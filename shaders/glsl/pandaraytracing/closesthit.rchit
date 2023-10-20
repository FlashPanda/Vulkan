#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;

vec3 lightPos(-0.005,0.04,1.98);	// world position of light

struct Vertex {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec3 color;
	vec4 tangent;
};

Vertex unpack(uint index)
{
	
}

void main()
{
  const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
  hitValue = barycentricCoords;
}
