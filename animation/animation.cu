//nvcc main.cu -I /usr/include/python3.10 -I /usr/lib/python3.10/site-packages/numpy/core/include/ -lpython3.10 -lboost_iostreams -L/usr/lib/python3.10/config-3.10m-x86_64-linux-gnu/
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <vector_functions.h>
#include <matplot/matplot.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <string>

#define num_of_particles 1000
#define k_boltzman  1.38 * 6.022e23 //1.38e-23 * 6.022e46
#define time_step 1e-11
#define final_time 1e-8
#define domain 1e4
#define protein_radius 100
#define protein_mass 10000
#define water_radius 3
#define water_mass 18
#define i_temp 300

struct Particle 
{
  float3 position;
  float3 velocity;
  float radius;
  float mass;
};

struct Initial_Conditions 
{
  int N;
  float tempereture;
  float s_radius, b_radius;
  float s_mass, b_mass;
  float2 xdomain, ydomain, zdomain;
  float dt;
  float tf;
};

__host__ __device__ inline float3 operator/(const float3 &a, float b) {
  return {a.x / b, a.y / b, a.z / b};
}
__host__ __device__ float dot(float3 a, float3 b)
{
  return (a.x*b.x + a.y*b.y + a.z*b.z);
}
__host__ __device__ float3 normalize(float3 v)
{
  float magnitude = sqrtf(dot(v,v));
  return v/magnitude;
}
__host__ __device__ inline float3 operator*(const float3 &a, float b) {
  return {a.x * b, a.y * b, a.z * b};
}
__host__ __device__ inline float3 operator*(float b, const float3 &a) {
  return {a.x * b, a.y * b, a.z * b};
}

__host__ __device__ float length(const float3 &a)
{
  return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}
__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b )
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b )
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}



__global__ void initial_conditions(Particle *particles, Initial_Conditions ic)
{
  curandState_t state;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i > ic.N)
    return;
  curand_init(clock64(), i, 0, &state);
  if(i == 0)
  {
    particles[i].position = make_float3(ic.xdomain.x + ic.xdomain.y/2.f,ic.xdomain.x + ic.ydomain.y/2.f,ic.xdomain.x +ic.zdomain.y/2.f);
    float sigma = sqrtf(k_boltzman * ic.tempereture / ic.b_mass);
    float speed = curand_normal(&state) * sigma;
    float3 direction = normalize(make_float3(curand_uniform(&state),curand_uniform(&state),curand_uniform(&state)));
    particles[i].velocity = direction * speed;
    particles[i].mass = ic.b_mass;
    particles[i].radius = ic.b_radius;

  } else
  {
    particles[i].position = make_float3(ic.xdomain.x + (ic.xdomain.y - ic.xdomain.x)*curand_uniform(&state),
    ic.ydomain.x + (ic.ydomain.y - ic.ydomain.x)*curand_uniform(&state),
    ic.zdomain.x + (ic.zdomain.y - ic.zdomain.x)*curand_uniform(&state));

    float sigma = sqrtf(k_boltzman * ic.tempereture / ic.s_mass);
    float speed = curand_normal(&state) * sigma;
    float3 direction = normalize(make_float3(curand_uniform(&state),curand_uniform(&state),curand_uniform(&state)));
    particles[i].velocity = direction * speed;
    particles[i].mass = ic.s_mass;
    particles[i].radius = ic.s_radius;
  }
}


__device__ void boundary_check(Particle& p, const Initial_Conditions& ic)
{
  float3 domain_size = make_float3(ic.xdomain.y - ic.xdomain.x, 
                                   ic.ydomain.y - ic.ydomain.x, 
                                   ic.zdomain.y - ic.zdomain.x);
  if (p.position.x > ic.xdomain.y)
  {
    p.position.x = ic.xdomain.x + fmodf(p.position.x - ic.xdomain.y, domain_size.x);
  }
  else if (p.position.x < ic.xdomain.x)
  {
    p.position.x = ic.xdomain.y - fmodf(ic.xdomain.x - p.position.x, domain_size.x);
  }

  if (p.position.y > ic.ydomain.y)
  {
    p.position.y = ic.ydomain.x + fmodf(p.position.y - ic.ydomain.y, domain_size.y);
  }
  else if (p.position.y < ic.ydomain.x)
  {
    p.position.y = ic.ydomain.y - fmodf(ic.ydomain.x - p.position.y, domain_size.y);
  }

  if (p.position.z > ic.zdomain.y)
  {
    p.position.z = ic.zdomain.x + fmodf(p.position.z - ic.zdomain.y, domain_size.z);
  }
  else if (p.position.z < ic.zdomain.x)
  {
    p.position.z = ic.zdomain.y - fmodf(ic.zdomain.x - p.position.z, domain_size.z);
  }
}

__global__ void detectCollisions(Particle *d_particles, int num_particles, float3 *d_collisions)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_particles) return;

    for(int i = 0; i < num_particles; i++)
    {
        if(i == tid)
            continue;
        float sum_radius = d_particles[tid].radius + d_particles[i].radius;
        float3 delta = d_particles[tid].position - d_particles[i].position;
        float distance = length(delta);

        if(distance <= sum_radius)
        {
            float3 normal = normalize(delta);
            float3 relative_velocity = d_particles[tid].velocity - d_particles[i].velocity;
            float impulse = 2.f * d_particles[i].mass* dot(relative_velocity, normal) / (d_particles[i].mass + d_particles[tid].mass);
            float3 impulse_vector = normal * impulse;
            //if(tid==0) printf("%e\n",impulse);
            d_collisions[tid] = d_collisions[tid] - impulse_vector;
            //if(tid == 0) printf("i %e %e %e \n", d_collisions[tid].x, d_collisions[tid].y,d_collisions[tid].z);

        }
    }
}

__global__ void processCollisions(Particle *d_particles, Initial_Conditions ic, float3 *d_collisions)
{
int tid = threadIdx.x + blockIdx.x * blockDim.x;
if (tid >= ic.N) return;
//if(tid == 0 && d_collisions[tid].x != 0 && d_collisions[tid].y != 0 && d_collisions[tid].z != 0)
  //printf("%e %e %e \n", d_collisions[tid].x, d_collisions[tid].y,d_collisions[tid].z);
d_particles[tid].velocity = d_particles[tid].velocity + d_collisions[tid];
//if(tid == 0) printf("v %e %e %e \n", d_particles[tid].velocity.x, d_particles[tid].velocity.y,d_particles[tid].velocity.z);
d_collisions[tid] = make_float3(0,0,0);
}

__global__ void updatePosition(Particle *d_particles, Initial_Conditions ic)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= ic.N) return;
  d_particles[tid].position = d_particles[tid].position + d_particles[tid].velocity * ic.dt;
  
  boundary_check(d_particles[tid], ic);
}


void runSimulation(Initial_Conditions ic)
{
  // allocate memory for particles on the device
  Particle* d_particles;
  cudaMalloc(&d_particles, ic.N * sizeof(Particle));

  // allocate memory for collisions on the device
  float3* d_collisions;
  cudaMalloc(&d_collisions, ic.N * sizeof(float3));

  // create a stream for the detectCollisions kernel
  cudaStream_t detectCollisionsStream;
  cudaStreamCreate(&detectCollisionsStream);

  // open a file to write the output
  std::ofstream output_file("data.txt");

  // launch the detectCollisions kernel
  int block_size = 256;
  int grid_size = (ic.N + block_size - 1) / block_size;
  initial_conditions<<<grid_size, block_size, 0, detectCollisionsStream>>>(d_particles, ic);
  cudaStreamSynchronize(detectCollisionsStream);

  float t = 0;
  while(t < final_time)
  {
    detectCollisions<<<grid_size, block_size, 0, detectCollisionsStream>>>(d_particles, ic.N, d_collisions);
    cudaStreamSynchronize(detectCollisionsStream);

    // create a stream for the processCollisions kernel
    cudaStream_t processCollisionsStream;
    cudaStreamCreate(&processCollisionsStream);

    // launch the processCollisions kernel
    processCollisions<<<grid_size, block_size, 0, processCollisionsStream>>>(d_particles, ic, d_collisions);
    cudaStreamSynchronize(processCollisionsStream);

    // create a stream for the updatePosition kernel
    cudaStream_t updatePositionStream;
    cudaStreamCreate(&updatePositionStream);

    // update the position of each particle due to velocity
    updatePosition<<<grid_size, block_size, 0, updatePositionStream>>>(d_particles, ic);
    cudaStreamSynchronize(updatePositionStream);

    t += time_step;
    // output the position of each particle in the file
    for (int i = 0; i < num_of_particles; i++) {
      Particle particle;
      cudaMemcpy(&particle, d_particles + i, sizeof(Particle), cudaMemcpyDeviceToHost);
      output_file << t << ' ' << particle.position.x << ' ' << particle.position.y << ' ' << particle.position.z << '\n';
    }

    // destroy the streams
    cudaStreamDestroy(processCollisionsStream);
    cudaStreamDestroy(updatePositionStream);
  }
  // close the output file
  output_file.close();

  // deallocate the memory on the device
  cudaFree(d_particles);
 

  cudaFree(d_collisions);

  // destroy the detectCollisions stream
  cudaStreamDestroy(detectCollisionsStream);
}


int main(void)
{
  Initial_Conditions ic{num_of_particles,
  i_temp,
  water_radius, protein_radius,
  water_mass, protein_mass,
  make_float2(0,domain), make_float2(0,domain), make_float2(0,domain),
  time_step, final_time};
  runSimulation(ic);
  return 0;
}