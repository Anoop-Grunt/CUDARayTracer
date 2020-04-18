#include "sphere.cuh"


sphere::sphere(vec3 center, float r)
    :center(center), radius(r)
{
}

sphere::~sphere()
{
}

__device__ bool sphere::hit(ray r, float t_min, float t_max, hit_record& rec) {

    vec3 oc = r.get_origin() - center;
    float a = dot(r.get_direction(), r.get_direction());
    float h = dot(oc, r.get_direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = h * h - a * c;
    if (discriminant > 0) {
        float temp = (-h - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.get_point_at_t(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;

        }

        temp = (-h + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.get_point_at_t(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }

    }

    return false;
}
