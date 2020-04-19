#include "sphere.cuh"


sphere::sphere(vec3 center, float r)
    :center(center), radius(r)
{
}

sphere::~sphere()
{
}

__device__ bool sphere::hit(ray r, float t_min, float t_max, sphere_hit_details& record) {

    vec3 oc = r.get_origin() - center;
    float a = dot(r.get_direction(), r.get_direction());
    float h = dot(oc, r.get_direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = h * h - a * c;
    if (discriminant > -0.f){

        // if both roots are real, the ray hits at two points, in which case there will be an incoming and outgoing ray
        //For the incoming ray we set  the normal in the same direction as (r(t) -cen)
        //but for outgoing ray we flip the normal, so that it points outwards from the sphere
        
        float temp = (-h - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            record.t = temp;
            record.p = r.get_point_at_t(record.t);
            record.normal = (record.p - center) / radius;
            vec3 outward_normal = (record.p - center) / radius;
            record.orient_normal(r, outward_normal);
            return true;

        }

        temp = (-h + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min) {
            record.t = temp;
            record.p = r.get_point_at_t(record.t);
            record.normal = (record.p - center) / radius;
            vec3 outward_normal = (record.p - center) / radius;
            record.orient_normal(r, outward_normal);
            return true;
        }

    }

    return false;
}
