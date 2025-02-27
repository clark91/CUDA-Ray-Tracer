#include "geometry.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

std::vector<tri> parseObj(std::string file, Material material, Vector3f offset){
  std::vector<Vector3f> vertices;
  std::vector<tri> tris;

  std::ifstream objFile(file);

  std::string line;
  while (getline(objFile, line)){
    if (line[0] == 'f'){
      std::vector<std::string> tokens = parseLine(line, ' ');
      std::vector<std::string> triIds;
      for (std::string t: tokens){
        std::vector<std::string> triId = parseLine(t, '/');
        triIds.push_back(triId[0]);
      }

      for (int i = 0; i < tokens.size() -2; i++){
        tri tmpTri;
        tmpTri.a = vertices[std::stoi(triIds[0]) - 1];
        tmpTri.b = vertices[std::stoi(triIds[i+1]) - 1];
        tmpTri.c = vertices[std::stoi(triIds[i+2]) - 1];
        tmpTri.updateEdges();
        tmpTri.material = material;

        tris.push_back(tmpTri);
      }
      

    } else if(line[0] == 'v' && line[1] == ' '){
      Vector3f vert;
      std::vector<std::string> tokens = parseLine(line, ' ');

      float scale = 1.f;
      vert.x = std::stof(tokens[0]) * scale;
      vert.y = std::stof(tokens[1]) * scale;
      vert.z = std::stof(tokens[2]) * scale;

      vert = vert + offset;

      vertices.push_back(vert);
    }
  }

  std::cout << "FILE READ\n";
  std::cout << "Vertices: " << vertices.size() << "\n";
  std::cout << "Polygons: " << tris.size() << "\n";

  objFile.close();
  return tris;
}

std::vector<std::string> parseLine(std::string input, char delim){
  std::stringstream ss(input);
  std::vector<std::string> tokens;
  std::string token;

  while (getline(ss, token, delim)){
    if (token != "f" && token != "v"){
      tokens.push_back(token);
    }
  }
  return tokens;
}

/*int main(int argc, char const *argv[])
{
  parseObj();
  return 0;
}*/
