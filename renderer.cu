#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include "geometryCuda.hpp"

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1920
#define RAY_DEPTH 3

__host__ __device__ void swap(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ Vector3f reflect(Vector3f I, Vector3f N) {
    return (I - N * 2.f * I.dot(N));
}

__device__ Vector3f refract(Vector3f &I, Vector3f &N, const float &refractive_index) {
    float cosi = fmax(-1.f, fmin(1.f, I.dot(N)));
    float etai = 1, etat = refractive_index;

    Vector3f n = N;
    if (cosi < 0) {
        cosi = -cosi;
        swap(etai, etat); n = -N;
    }
    float eta = etai / etat;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? Vector3f(0, 0, 0) : I * eta + n * (eta * cosi - sqrtf(k));
}

__device__ HitInfo sceneHit(Vector3f &orig, Vector3f &dir, tri *tris, int numTris) {
    HitInfo closest;
    closest.distance = MAXFLOAT;
    for (int i = 0; i < numTris; i++) {
        HitInfo current = tris[i].intersectsRay(orig, dir);
        if (current.didHit && current.distance < closest.distance) {
            closest = current;
        }
    }
    return closest;
}

__device__ Vector3f castRay(Vector3f &orig, Vector3f &dir, tri *tris, int numTris, Light *lights, int numLights, size_t depth = 0) {
    HitInfo closestTri;
    closestTri.distance = 999999.f;
    closestTri.didHit = false;

    for (int i = 0; i < numTris; i++) {
        HitInfo hitInfo = tris[i].intersectsRay(orig, dir);
        if (hitInfo.didHit && hitInfo.distance < closestTri.distance) {
            closestTri = hitInfo;
        }
    }

    if (!closestTri.didHit || depth > RAY_DEPTH) {
        return Vector3f(0.2, 0.7, 0.8);
    }

    Vector3f reflect_dir = reflect(dir, closestTri.normal).normalize();
    Vector3f reflect_orig = reflect_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal*1e-3 : closestTri.position + closestTri.normal*1e-3;
    Vector3f reflect_color = castRay(reflect_orig, reflect_dir, tris, numTris, lights, numLights, depth + 1);

    Vector3f refract_dir = refract(dir, closestTri.normal, closestTri.material.refractive_index).normalize();
    Vector3f refract_orig = refract_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal * 1e-3 : closestTri.position + closestTri.normal * 1e-3;
    Vector3f refract_color = castRay(refract_orig, refract_dir, tris, numTris, lights, numLights, depth + 1);

    float diffuse_light_intensity = 0, specular_light_intensity = 0;

    for (int i = 0; i < numLights; i++) {
        Vector3f light_dir = (lights[i].position - closestTri.position).normalize();
        diffuse_light_intensity += lights[i].intensity * fmax(0.f, closestTri.normal.dot(light_dir));

        specular_light_intensity += powf(fmax(0.f, -reflect(-light_dir, closestTri.normal).dot(dir)), closestTri.material.specular_exponent)*lights[i].intensity;

        float light_distance = (lights[i].position - closestTri.position).magnitude();
        Vector3f shadow_orig = light_dir.dot(closestTri.normal) < 0 ? closestTri.position - closestTri.normal * 1e-3 : closestTri.position + closestTri.normal * 1e-3;

        HitInfo hitInfo = sceneHit(shadow_orig, light_dir, tris, numTris);
        if (hitInfo.didHit && (hitInfo.position - shadow_orig).magnitude() < light_distance) {
            continue;
        }
    }

    return closestTri.material.color * diffuse_light_intensity * closestTri.material.albedo[0] + Vector3f(1, 1, 1) * specular_light_intensity * closestTri.material.albedo[1] + reflect_color * closestTri.material.albedo[2] + refract_color * closestTri.material.albedo[3];
}

__global__ void renderKernel(Vector3f *framebuffer, int width, int height, float fov, tri *tris, int numTris, Light *lights, int numLights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= width || j >= height) return;

    Vector3f cameraPos = Vector3f(0, 0, 0);
    float x = (2 * (i + 0.5) / (float)width - 1) * tan(fov / 2.) * width / (float)height;
    float y = -(2 * (j + 0.5) / (float)height - 1) * tan(fov / 2.);
    Vector3f dir = Vector3f(x, y, -1).normalize();
    framebuffer[i + j * width] = castRay(cameraPos, dir, tris, numTris, lights, numLights);
}

void render(std::vector<tri> &tris, std::vector<Light> &lights) {
    const int width = SCREEN_WIDTH;
    const int height = SCREEN_HEIGHT;
    const float fov = (45.f / 180.f) * M_PI;

    std::vector<Vector3f> framebuffer(width * height);

    Vector3f *d_framebuffer;
    tri *d_tris;
    Light *d_lights;

    cudaMalloc(&d_framebuffer, width * height * sizeof(Vector3f));
    cudaMalloc(&d_tris, tris.size() * sizeof(tri));
    cudaMalloc(&d_lights, lights.size() * sizeof(Light));

    cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(tri), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lights, lights.data(), lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

    size_t stackSize = 4096 * 4096 * 4; 
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    renderKernel<<<gridSize, blockSize>>>(d_framebuffer, width, height, fov, d_tris, tris.size(), d_lights, lights.size());

    cudaMemcpy(framebuffer.data(), d_framebuffer, width * height * sizeof(Vector3f), cudaMemcpyDeviceToHost);

    cudaFree(d_framebuffer);
    cudaFree(d_tris);
    cudaFree(d_lights);

    std::cout << "Rendering Complete\n";

    std::ofstream ofs;
    ofs.open("./out.ppm");
    ofs << "P6\n" << width << " " << height << "\n255\n";

    for (size_t i = 0; i < height * width; i++) {
        for (size_t j = 0; j < 3; j++) {
            ofs << (char)(255 * std::max(0.f, std::min(1.f, framebuffer[i][j])));
        }
    }
}

int main() {
    Material ivory(Vector3f(0.6, 0.3, 0.1), 50., Vector4f(0.4, 0.4, 0.3,0.0),1.0);
    Material red_rubber(Vector3f(0.9, 0.1, 0.0), 10., Vector4f(0.3, 0.1, 0.1, 0.0),1.0);
    Material mirror(Vector3f(1.0, 1.0, 1.0), 9999., Vector4f(1.0, 1.0, 1.0, 0.0),1.0);
    Material glass(Vector3f(0.6, 0.7, 0.8), 125., Vector4f(0.0, 0.5, 0.1, 0.8), 1.5);

    std::vector<tri> triangles;

    std::vector<tri> monkey = parseObj("monkey.obj", ivory, Vector3f(0., 0., -4.));
    triangles.insert(std::end(triangles), std::begin(monkey), std::end(monkey));

    std::vector<tri> teapot = parseObj("teapot.obj", glass, Vector3f(0., -2., -10.));
    //triangles.insert(std::end(triangles), std::begin(teapot), std::end(teapot));

    std::vector<tri> teapot2 = parseObj("teapot.obj", red_rubber, Vector3f(0., -2, -20.));
    //triangles.insert(std::end(triangles), std::begin(teapot2), std::end(teapot2));

    std::vector<Light> lights;
    lights.push_back(Light(Vector3f(30, 50, -25), 5.0f));

    auto beg = std::chrono::high_resolution_clock::now();
    render(triangles, lights);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - beg);

    std::cout << "In " << duration.count() << " seconds\n";
    return 0;
}