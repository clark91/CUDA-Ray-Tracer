#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<cmath>
#include<thread>
#include<chrono>
#include "geometry.hpp"

#define SCREEN_WIDTH 200
#define SCREEN_HEIGHT 200
#define RAY_DEPTH 3

Vector3f reflect (Vector3f I, Vector3f N){
  return (I - N * I.dot(N) * 2.f);
}

Vector3f refract(Vector3f &I, Vector3f &N, const float &refractive_index){
  float cosi = std::max(-1.f, std::min(1.f, I.dot(N)));
  float etai = 1, etat = refractive_index;

  Vector3f n = N;
  if (cosi < 0){
    cosi = -cosi;
    std::swap(etai, etat); n = -N;
  }
  float eta = etai / etat;
  float k = 0 - eta*eta*(1 - cosi*cosi);
  return k < 0 ? Vector3f(0,0,0) : I*eta + n*(eta * cosi - sqrtf(k));
}

HitInfo sceneHit(Vector3f &orig, Vector3f &dir, std::vector<tri> tris){
  HitInfo closest;
  closest.distance = MAXFLOAT;
  for(int i = 0; i < tris.size(); i++){
    HitInfo current = tris[i].intersectsRay(orig, dir);
    if(current.didHit && current.distance < closest.distance){ 
      closest = current;
    }
  }
  return closest;
}

Vector3f castRay (Vector3f &orig, Vector3f &dir, std::vector<tri> &tris, std::vector<Light> &lights, size_t depth = 0){
  HitInfo closestTri;
  closestTri.distance = 999999.f;
  closestTri.didHit = false;
  for (tri triangle : tris){
    HitInfo hitInfo = triangle.intersectsRay(orig, dir);

    if (hitInfo.didHit && hitInfo.distance < closestTri.distance){
      closestTri = hitInfo;
    }
  }

  if (!closestTri.didHit || depth > RAY_DEPTH){
    return Vector3f(0.2, 0.7, 0.8);
  }

  Vector3f reflect_dir = reflect(dir, closestTri.normal).normalize();
  Vector3f reflect_orig = reflect_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal*1e-3 : closestTri.position + closestTri.normal*1e-3;
  Vector3f reflect_color = castRay(reflect_orig, reflect_dir, tris, lights, depth + 1);

  Vector3f refract_dir = refract(dir, closestTri.normal, closestTri.material.refractive_index).normalize();
  Vector3f refract_orig = refract_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal*1e-3 : closestTri.position + closestTri.normal*1e-3;
  Vector3f refract_color = castRay(refract_orig, reflect_dir, tris, lights, depth+1);

  float diffuse_light_intensity = 0, specular_light_intensity = 0;

  for (size_t i = 0; i < lights.size(); i++){
    Vector3f light_dir = (lights[i].position - closestTri.position).normalize();
    diffuse_light_intensity += lights[i].intensity * std::max(0.f, closestTri.normal.dot(light_dir));

    specular_light_intensity += powf(std::max(0.f, -reflect(-light_dir, closestTri.normal).dot(dir)), closestTri.material.specular_exponent)*lights[i].intensity;

    float light_distance = (lights[i].position - closestTri.position).magnitude();
    Vector3f shadow_orig = light_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal * 1e-3 : closestTri.position + closestTri.normal*1e-3;
  
    HitInfo hitInfo = sceneHit(shadow_orig, light_dir, tris);
    if (hitInfo.didHit && (hitInfo.position - shadow_orig).magnitude() < light_distance){
      continue;
    }
  }

  return closestTri.material.color * diffuse_light_intensity * closestTri.material.albedo[0] + Vector3f(1,1,1) * specular_light_intensity * closestTri.material.albedo[1] + reflect_color*closestTri.material.albedo[2] + refract_color*closestTri.material.albedo[3];
  
}

void renderTile(int startX, int endX, int startY, int endY, std::vector<tri> &tris, std::vector<Light> &lights, std::vector<Vector3f> &framebuffer, int width, int height, float fov) {
  Vector3f cameraPos = Vector3f(0, 0, 0);

  for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
          float x = (2 * (i + 0.5) / (float)width - 1) * tan(fov / 2.) * width / (float)height;
          float y = -(2 * (j + 0.5) / (float)height - 1) * tan(fov / 2.);
          Vector3f dir = Vector3f(x, y, -1).normalize();
          framebuffer[i + j * width] = castRay(cameraPos, dir, tris, lights);
      }
  }
}

void render(std::vector<tri> &tris, std::vector<Light> &lights) {
  const int width = SCREEN_WIDTH;
  const int height = SCREEN_HEIGHT;
  const float fov = (45.f / 180.f) * M_PI;

  std::vector<Vector3f> framebuffer(width * height);

  int numThreads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;

  int tileSizeX = width / numThreads;
  int tileSizeY = height / numThreads;

  for (int y = 0; y < numThreads; y++) {
      for (int x = 0; x < numThreads; x++) {
          int startX = x * tileSizeX;
          int endX = (x == numThreads - 1) ? width : startX + tileSizeX;

          int startY = y * tileSizeY;
          int endY = (y == numThreads - 1) ? height : startY + tileSizeY;

          threads.push_back(std::thread(renderTile, startX, endX, startY, endY, std::ref(tris), std::ref(lights), std::ref(framebuffer), width, height, fov));
      }
  }

  for (auto &t : threads) {
      t.join();
  }

  std::cout << "Rendering Complete\n";

  std::ofstream ofs;
  ofs.open("./out.ppm");
  ofs << "P6\n" << width << " " << height << "\n255\n";

  for (size_t i = 0; i < height * width; i++){
    for (size_t j = 0; j< 3; j++){
      ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
    }
  }


}

int main(){

  Material ivory(Vector3f(0.6, 0.3, 0.1), 50., Vector4f(0.4,0.4,0.3));
  Material red_rubber(Vector3f(0.9,0.1,0.0), 10., Vector4f(0.3,0.1,0.1));
  Material mirror(Vector3f(0.0,10.0,0.8), 1425., Vector4f(1.0,1.0,1.0));
  Material glass(Vector3f(0.6, 0.7, 0.8), 125., Vector4f(0.0,  0.5, 0.1, 0.8), 1.5);

  //std::vector<tri> triangles;
  
  //triangles.push_back(tri(Vector3f(8.0,-1.5,-4.9), Vector3f(-1,1.5,-5), Vector3f(1,0,-5), red_rubber)); 
  //triangles.push_back(tri(Vector3f(8.0,1.5,-5), Vector3f(-1,1.5,-5), Vector3f(1,0,-5), mirror)); 

  std::vector<tri> triangles;

  std::vector<tri> monkey = parseObj("monkey.obj", glass, Vector3f(0.,0.,-4.));
  triangles.insert(std::end(triangles), std::begin(monkey), std::end(monkey));

  std::vector<tri> teapot = parseObj("teapot.obj", red_rubber, Vector3f(0.,-1.,-7.));
  triangles.insert(std::end(triangles), std::begin(teapot), std::end(teapot));

  //std::vector<tri> cube = parseObj("cube.obj");
  //triangles.insert(std::end(triangles), std::begin(cube), std::end(cube));

  std::vector<Light> lights;
  lights.push_back(Light(Vector3f( 30, 50, -25),5.0f));

  auto beg = std::chrono::high_resolution_clock::now();
  render(triangles, lights);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - beg);

  std::cout << "In " << duration.count() << " seconds\n";
  return 0;
}