#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec2 attribs;

layout(binding = 2, set = 0) uniform UBO
{
	mat4 viewInverse;
	mat4 projInverse;
	int vertexSize;
	int materialSize;
	int primitiveSize;
} ubo;
layout(binding = 3, set = 0) buffer Vertices { vec4 v[]; } vertices;
layout(binding = 4, set = 0) buffer Indices { uint i[]; } indices;
layout(binding = 5, set = 0) buffer Materials {vec4 m[]; } materials;
layout(binding = 6, set = 0) buffer Primitives {uint p[];} primitives;

struct Vertex {
	vec3 pos;
	vec3 normal;
	vec2 uv;
	vec3 color;
	vec4 tangent;
	float pad;
};

struct ShadeMaterial
{
	vec4		baseColorFactor;
	uint		baseColorTextureIndex;
	float		metallicFactor;
	float		roughnessFactor;
	float		pad;	// 补足对齐用
};

struct Primitive {
	uint firstIndex;
	uint indexCount;
	int materialIndex;
};

const float PI = 3.14159265359;

Vertex unpack(uint index)
{
	// Unpack the vertices from the SSBO using the glTF vertex structure
	// The multiplier is the size of the vertex divided by four float components (16 bytes)
	const int m = ubo.vertexSize / 16;
	
	vec4 d0 = vertices.v[m * index + 0];
	vec4 d1 = vertices.v[m * index + 1];
	vec4 d2 = vertices.v[m * index + 2];
	vec4 d3 = vertices.v[m * index + 3];
	
	Vertex v;
	v.pos = d0.xyz;
	v.normal = vec3(d0.w, d1.x, d1.y);
	v.uv = vec2(d1.z, d1.w);
	v.color = d2.xyz;
	v.tangent = vec4(d2.w, d3.x, d3.y, d3.z);
	
	return v;
}

ShadeMaterial unpackMaterial(uint index)
{
	const int t = ubo.materialSize / 16;
	ShadeMaterial material;
	material.baseColorFactor = materials.m[t * index + 0];
	material.baseColorTextureIndex = floatBitsToInt(materials.m[t * index + 1][0]);
	material.metallicFactor = materials.m[t * index + 1][1];
	material.roughnessFactor = materials.m[t * index + 1][2];
	return material;
}

Primitive unpackPrimitive(uint index)
{
	const int t = ubo.primitiveSize / 4;
	
	Primitive primitive;
	primitive.firstIndex = primitives.p[t * index + 0];
	primitive.indexCount = primitives.p[t * index + 1];
	primitive.materialIndex = floatBitsToInt(uintBitsToFloat(primitives.p[t * index + 2]));
	return primitive;
}

// Cook-Torrance BRDF
// From https://zhuanlan.zhihu.com/p/152226698 and https://zhuanlan.zhihu.com/p/158025828
// Render equation: $L_o=\int_{\Omega}f_rL_i\cos\theta_i\mathrm{d}\omega_i$
// $f_r=k_df_{lambert}+k_sf_{cook-torrance}$
// use metallic and albedo to determine $F_0$, $F_0=mix(vec3(0.04), albedo, metallic)$
// $F\approx F_0+(1-F_0)(1-\cos\theta_i)^5$
// $k_d=(1-F)(1-metallic)$
// finally, $f=(1-metallic)\frac{albedo}{\pi}+\frac{D F G}{4\cos\theta_i \cos\theta_o}$

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

void main()
{
	// world position of light
	vec3 lightPos = vec3(-0.005, 0.04, 1.2);
	
	ivec3 index = ivec3(indices.i[3 * gl_PrimitiveID], indices.i[3 * gl_PrimitiveID + 1], indices.i[3 * gl_PrimitiveID + 2]);
	
	// get vertex info
	Vertex v0 = unpack(index.x);
	Vertex v1 = unpack(index.y);
	Vertex v2 = unpack(index.z);
	
	
	// Interpolate normal
	const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	vec3 normal = normalize(v0.normal * barycentricCoords.x + v1.normal * barycentricCoords.y + v2.normal * barycentricCoords.z);
	vec3 hitPos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
	vec3 hitPos_world = vec3(gl_ObjectToWorldEXT * vec4(hitPos, 1.0));
	
	// pbr rendering
	vec3 N = normal;
	vec3 V = normalize((ubo.viewInverse * vec4(0,0,0,1)).xyz - hitPos_world);
	
	// light
	float intensity = 10;
	vec4 lightColor = vec4(1, 1, 1, 1);
	vec3 L = normalize(lightPos - hitPos_world);
	vec3 H = normalize( V + L);
	float NDotL = max(dot(N, L), 0.0);
	
	float distance = length(lightPos  - hitPos_world);
	float attenuation = 1.0 / ((0.6 * distance * distance) + 1.0);
	vec3 radiance = lightColor.xyz * intensity * attenuation;
	
	vec3 albedo = vec3(0.5, 0.5, 0.5);
	float metallic = 0.1;
	float roughness = 0.4;
	
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, albedo, metallic);
	
	vec3 F = fresnelSchlick(max(dot(H, N), 0.0), F0);
	float NDF = DistributionGGX(N, H, roughness);
	float G = GeometrySmith(N, V, L, roughness);
	
	vec3 Nominator = NDF * G * F;
	float Denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
	vec3 Specular = Nominator / Denominator;
	
	vec3 kd = (1 - F)* (1 - metallic);
	
	vec3 Lo = (kd * albedo / PI + Specular) * radiance * NDotL;
	
	// gamma correction
	vec3 color = Lo;
	color = pow(color, vec3(1.0 / 2.2));
	
	hitValue = color;
}
