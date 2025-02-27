#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

struct Vector2f {
    float x, y;
    __host__ __device__ Vector2f(float a = 0, float b = 0) : x(a), y(b) {}

    __host__ __device__ float& operator[](size_t index) {
        //if (index > 1) throw std::out_of_range("Index out of range");
        return (index == 0) ? x : y;
    }
};

struct Vector3f {
    float x, y, z;
    __host__ __device__ Vector3f(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}

    __host__ __device__ float& operator[](size_t index) {
        //if (index > 2) throw std::out_of_range("Index out of range");
        return (index == 0) ? x : (index == 1) ? y : z;
    }

    __host__ __device__ Vector3f normalize() const {
        float magnitude = sqrtf(x * x + y * y + z * z);
        if (magnitude == 0) return *this;
        float x2 = x / magnitude;
        float y2 = y / magnitude;
        float z2 = z / magnitude;
        return Vector3f(x2, y2, z2);
    }

    __host__ __device__ float magnitude() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ Vector3f operator-(Vector3f sub) const {
        float x2 = x - sub[0];
        float y2 = y - sub[1];
        float z2 = z - sub[2];
        return Vector3f(x2, y2, z2);
    }

    __host__ __device__ Vector3f operator+(Vector3f sub) const {
        float x2 = x + sub[0];
        float y2 = y + sub[1];
        float z2 = z + sub[2];
        return Vector3f(x2, y2, z2);
    }

    __host__ __device__ Vector3f cross(Vector3f b) const {
        return Vector3f((y * b[2]) - (z * b[1]), (z * b[0]) - (x * b[2]), (x * b[1]) - (y * b[0]));
    }

    __host__ __device__ float dot(Vector3f b) const {
        return (x * b[0] + y * b[1] + z * b[2]);
    }

    __host__ __device__ Vector3f operator*(float multiplier) const {
        return Vector3f(x * multiplier, y * multiplier, z * multiplier);
    }

    __host__ __device__ Vector3f operator-() const {
        return Vector3f(-x, -y, -z);
    }
};

struct Vector4f {
    float x, y, z, w;
    __host__ __device__ Vector4f(float a = 0, float b = 0, float c = 0, float d = 0) : x(a), y(b), z(c), w(d) {}

    __host__ __device__ float& operator[](size_t index) {
        //if (index > 3) throw std::out_of_range("Index out of range");
        return (index == 0) ? x : (index == 1) ? y : (index == 2) ? z : w;
    }
};

struct Light {
    __host__ __device__ Light(const Vector3f &p, const float &i) : position(p), intensity(i) {}
    Vector3f position;
    float intensity;
};

struct Material {
    Vector3f color;
    Vector4f albedo;
    float specular_exponent;
    float refractive_index;

    __host__ __device__ Material(Vector3f c, float s, Vector4f a, float r = 1) : color(c), specular_exponent(s), albedo(a), refractive_index(r) {}

    __host__ __device__ Material() {}
};

struct HitInfo {
    Vector3f position, normal;
    float distance;
    Material material;
    bool didHit;

    __host__ __device__ HitInfo() {}
};

struct tri {
    Vector3f a, b, c, E1, E2;
    Material material;

    __host__ __device__ tri(Vector3f x, Vector3f y, Vector3f z, Material m) : a(x), b(y), c(z), material(m) {
        E1 = b - a;
        E2 = c - a;
    }

    __host__ __device__ tri() {}

    __host__ __device__ void updateEdges() {
        E1 = b - a;
        E2 = c - a;
    }

    __host__ __device__ HitInfo intersectsRay(Vector3f &orig, Vector3f &dir) const {
        HitInfo hitInfo;
        hitInfo.material = material;

        Vector3f h = dir.cross(E2);
        float det = E1.dot(h);

        if (std::abs(det) < 1e-8) {
            hitInfo.didHit = false;
            return hitInfo;
        }

        float invDet = 1.0f / det;

        Vector3f s = orig - a;
        float u = invDet * s.dot(h);

        if (u < 0.0f || u > 1.0f) {
            hitInfo.didHit = false;
            return hitInfo;
        }

        Vector3f q = s.cross(E1);
        float v = invDet * dir.dot(q);

        if (v < 0.0f || u + v > 1.0f) {
            hitInfo.didHit = false;
            return hitInfo;
        }

        float t = invDet * E2.dot(q);

        if (t <= 0) {
            hitInfo.didHit = false;
            return hitInfo;
        }

        hitInfo.didHit = true;
        hitInfo.distance = t;
        hitInfo.normal = E2.cross(E1).normalize();
        hitInfo.position = orig + (dir * t);
        return hitInfo;
    }
};

std::vector<tri> parseObj(std::string file, Material material = Material(Vector3f(0.9, 0.1, 0.0), 10., Vector4f(0.3, 0.1, 0.1, 0.0)), Vector3f offset = Vector3f(0, 0, 0), float scale = 1.f);

std::vector<std::string> parseLine(std::string input, char delim);