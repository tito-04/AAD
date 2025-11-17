MARQUES
marques2222
Online

TiTo — 11/15/25, 10:31 PM
mostra
que eu resolvo
MARQUES — 11/15/25, 10:32 PM
aad27@banana:~$ make cuda_miner 
nvcc -arch=sm_75 --compiler-options -O2,-Wall -I/usr/local/cuda/include --cubin miner_kernel.cu -o miner_kernel.cubin
cc -march=native -Wall -Wshadow -Werror -O3 cuda_miner.c -o cuda_miner -lcuda
In file included from cuda_miner.c:27:0:
aad_cuda_utilities.h: In function ‘initialize_cuda’:
aad_cuda_utilities.h:180:3: error: passing argument 2 of ‘cuCtxCreate_v2’ makes integer from pointer without a cast [-Werror=int-conversion]
Expand
message.txt
3 KB
TiTo — 11/15/25, 10:32 PM
Vai a linha 180 do cuda utilities e elimina o parametro NULL
E daquela situação que mandei ao professor
MARQUES — 11/15/25, 10:33 PM
okapa
ja deu esta com2330M de velocidade
TiTo — 11/15/25, 10:33 PM
OK
MARQUES — 11/15/25, 10:34 PM
juro que nao entendo este fica com menos velocidade aqui e mais na 4070?
TiTo — 11/15/25, 10:34 PM
Nao sei brpp
MARQUES — 11/15/25, 10:34 PM
ninguem sabe so o tos
ze fiz aquela alteracao do while(pos<54) e nao deu erro nenhum
agora aumentou apenas 50M, queres que te envio o ficheiro com a mudanca?
TiTo — 11/15/25, 10:39 PM
Tambem consegui esse aumento
Espera deixa competar esta melhoria
MARQUES — 11/15/25, 10:40 PM
mas com a cena do while?
TiTo — 11/15/25, 10:41 PM
Nao
MARQUES — 11/15/25, 10:42 PM
junta as duas cenas entao
btw nesse codigo que enviaste podes colocar o     cd.grid_dim_x = 65536;  a 131072
TiTo — 11/15/25, 10:44 PM
Sim ja o fiz
CALMA BELHOTE
MARQUES — 11/15/25, 10:44 PM
e existe uma cena no ficheiro aad_sha1.h que podes testar
logo no inicio experimenta colocar esta linha nesse ficheiro a 256 #define RECOMENDED_CUDA_BLOCK_SIZE  128
MARQUES — 11/15/25, 10:45 PM
NAO TARDA COMEÇA O UFC
TiTo — 11/15/25, 10:47 PM
Same hit
shit
MARQUES — 11/15/25, 10:48 PM
ok isso era uma sorte se mudasse alguma cena e as outras cenas
TiTo — 11/15/25, 10:49 PM
Mano que estrano o que tenho agra começa no 4950 mas depois é sempre a cair
Ja vou no 4858
MARQUES — 11/15/25, 10:50 PM
pois é isso que nao faz sentido a tua GPU esta a aquecer demasiado
TiTo — 11/15/25, 10:50 PM
Deve ser yha
MARQUES — 11/15/25, 10:50 PM
sopra
TiTo — 11/15/25, 10:50 PM
Ja vai nos 4848
Daqui a bocado esta no 0 hahaha
MARQUES — 11/15/25, 10:51 PM
e nem esta a cair muito aquele codigo inicial que mandei ia dos 3400M para 3000
bue rapido
Btw uma cena que temos de levar em conta é que se os outros tem graficas melhores vao automaticamente ter velocidades melhores, convem saber nao só as velocidades que eles têm mas tambem as graficas que estao a usar
TiTo — 11/15/25, 10:55 PM
Sim  temo que saber o que estão a ter na banana 
MARQUES — 11/15/25, 10:55 PM
exato
mas a cena é que como vimos nao é bem a mesma coisa, um codigo que aumenta performance na banana nao necessariamente aumenta performance na GPU do pc
TiTo — 11/15/25, 10:56 PM
Pois...
MARQUES — 11/15/25, 10:57 PM
chegaste a testar aquela cena do while e o gride size? 
para ir ver UFC
TiTo — 11/15/25, 10:57 PM
SIm
Yha amanha ha mais i guess
estou preso nos 4950
MARQUES — 11/15/25, 10:58 PM
fds e deu o mesmo
aquela cena do while devia ter aumentado mais
mas ok fica assim
TiTo — 11/15/25, 10:58 PM
Mas nao consigo correr mano
Da me eror na função do xor
MARQUES — 11/15/25, 10:58 PM
nah tem de dar
TiTo — 11/15/25, 10:58 PM
logo no inicio do while
MARQUES — 11/15/25, 10:58 PM
usa isto
//
// Ficheiro: miner_kernel.cu
//
// Kernel CUDA de mineração de DETI coins de alto desempenho.
//
// Arquitetura:
Expand
message.txt
6 KB
TiTo — 11/15/25, 10:59 PM
nao reconhece estou te a dizer
TiTo — 11/15/25, 10:59 PM
Piorou bue a usar este
MARQUES — 11/15/25, 11:00 PM
bro tens o GPU estragado
TiTo — 11/15/25, 11:00 PM
Passou para 3600
hahahahah
esta tudo fdd
MARQUES — 11/15/25, 11:00 PM
nao pode
estamos a fazer alguma coisa mal a correr essas merda
nao pode estar a piorar assim
TiTo — 11/15/25, 11:00 PM
nao sei zezoca
MARQUES — 11/15/25, 11:00 PM
mas fica para amanha em call é mais facil
TiTo — 11/15/25, 11:01 PM
yha é isso
MARQUES — 11/15/25, 11:01 PM
nao podes falar agora a ver as lutas pois nao?
TiTo — 11/15/25, 11:03 PM
Agora não mais daqui a bocado posso se calhar
manda ai o link
MARQUES — 11/15/25, 11:04 PM
ok
calma que nao esta a funcionar
https://720pstream.lc/live/mma/islam-makhachev-lq
720pStream
Islam Makhachev LQ(SD) Live Streaming | 720pStream
Watch mma event Islam Makhachev LQ(SD) Live Streaming online at 720pStream. How to find mma Islam Makhachev LQ(SD) streams? Every game stream is here
TiTo — 11/15/25, 11:04 PM
............................................................
MARQUES — 11/15/25, 11:04 PM
so esta a dar o ultimo link para mim
fds esta em espanho
wtf
TiTo — 11/15/25, 11:05 PM
HAHAHAHA yha
wtf
E o som esta todo cortado tambe
MARQUES — 11/15/25, 11:06 PM
ok vai para o terceiro link
ja esta a dar
DC a comentar\
TiTo — 11/15/25, 11:06 PM
...
Que merda
MARQUES — 11/15/25, 11:06 PM
btw fala por whatsapp
TiTo — 11/15/25, 11:07 PM
esta bem
MARQUES
 started a call that lasted a few seconds. — Yesterday at 4:04 AM
MARQUES
 started a call that lasted 5 minutes. — Yesterday at 4:07 AM
TiTo
 started a call that lasted an hour. — Yesterday at 5:17 PM
MARQUES — Yesterday at 5:32 PM
Image
MARQUES — 3:48 PM
.
MARQUES — 3:48 PM
.
MARQUES — 3:48 PM
.
MARQUES — 4:00 PM
//
// Ficheiro: cuda_miner.c
//
// (Versão Otimizada com Pipeline Assíncrono e Granularidade N=16)
//
Expand
message.txt
10 KB
//
// Ficheiro: miner_kernel.cu
//
// (Versão final com Granularidade Aumentada - N=16)
//
Expand
message.txt
5 KB
// --- Ponto de Entrada Principal ---
int main(int argc, char argv[])
{
    signal(SIGINT, signal_handler);

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_seed = (unsigned int)(time(NULL) ^ (uintptr_t)&global_seed ^ (unsigned int)getpid() ^ (unsigned int)ts.tv_nsec);
    for(int i = 0; i < 10; i++) { rand_r(&global_seed); }
    global_counter_offset = ((u64_t)rand_r(&global_seed) << 32) | (u64_t)rand_r(&global_seed);
    clock_gettime(CLOCK_MONOTONIC, &ts);
    global_counter_offset ^= ((u64_t)ts.tv_nsec << 32) | (u64_t)ts.tv_sec;

    // Obter argumentos
    const charcustom_text = NULL;
    u64_t max_attempts = 0;
    if(argc > 1) custom_text = argv[1];
    if(argc > 2) max_attempts = strtoull(argv[2], NULL, 10);

    // Sanitizar texto
    char sanitized_text[64];
    if(custom_text != NULL) {
        size_t si = 0;
        for(size_t i = 0; custom_text[i] != '\0' && si + 1 < sizeof(sanitized_text); i++) {
            if(custom_text[i] == '\n') sanitized_text[si++] = '\b';
            else sanitized_text[si++] = custom_text[i];
        }
        sanitized_text[si] = '\0';
        run_cuda_miner(sanitized_text, max_attempts);
    } else {
        run_cuda_miner(NULL, max_attempts);
    }

    return 0;
//
// Ficheiro: cuda_miner.c
//
// (Versão Otimizada com Pipeline Assíncrono e Granularidade N=16)
//
Expand
message.txt
12 KB
TiTo — 4:16 PM
utilities
//
// TomÃ¡s Oliveira e Silva,  September 2025
//
// Arquiteturas de Alto Desempenho 2025/2026
//
// CUDA driver API stuff
Expand
message.txt
11 KB
cuda_miner
//
// Ficheiro: cuda_miner.c
//
// Anfitrião (Host) para o minerador CUDA de DETI coins
// (Versão Otimizada com Pipeline Assíncrono)
//
Expand
message.txt
13 KB
﻿
TiTo
tito_04
 
 
//
// TomÃ¡s Oliveira e Silva,  September 2025
//
// Arquiteturas de Alto Desempenho 2025/2026
//
// CUDA driver API stuff
//

#ifndef AAD_CUDA_UTILITIES
#define AAD_CUDA_UTILITIES

#include <cuda.h>


//
// data type used to store all CUDA related stuff
//

#define MAX_N_ARGUMENTS  4

typedef struct
{
  // input data
  int device_number;             // number of the device to initialize
  char *cubin_file_name;         // name of the cubin file to load (NULL if not needed)
  char *kernel_name;             // name of the CUDA kernel to load (NULL if not needed)
  u32_t data_size[2];            // the number of bytes of the two data arrays to allocate on the host and on the device (0 if not needed)
  // persistent data
  CUdevice     cu_device;        // the device yhandle
  char         device_name[256]; // the device name
  CUcontext    cu_context;       // the device context
  CUmodule     cu_module;        // the loaded cubin file contents
  CUfunction   cu_kernel;        // the pointer to the CUDA kernel
  CUstream     cu_stream;        // the command stream
  void        *host_data[2];     // the pointers to the host data
  CUdeviceptr  device_data[2];   // the pointers to the device data
  // launch kernel data
  unsigned int grid_dim_x;       // the number of grid blocks (in the X dimension, the only one we will use here)
  unsigned int block_dim_x;      // the number of threads in a block (in the X dimension, the only one we will use here, should be equal to RECOMENDED_CUDA_BLOCK_SIZE)
  int n_kernel_arguments;        // number of kernel arguments
  void *arg[MAX_N_ARGUMENTS];    // pointers to the kernel argument data

}
cuda_data_t;


//
// CU_CALL --- macro that should be used to call a CUDA driver API function and to test its return value
//
// it should be used to test the return value of calls such as
//   cuInit(device_number);
//   cuDeviceGet(&cu_device,device_number);
//
// in these cases, f_name is, respectively, cuInit and cuDeviceGet, and args is, respectively,
//   (device_number) and (&cu_device,device_number)
//

#define CU_CALL(f_name,args)                                                                                  \
  do                                                                                                          \
  {                                                                                                           \
    CUresult e = f_name args;                                                                                 \
    if(e != CUDA_SUCCESS)                                                                                     \
    { /* the call failed, terminate the program */                                                            \
      fprintf(stderr,"" # f_name "() returned %s (file %s, line %d)\n",cu_error_string(e),__FILE__,__LINE__); \
      exit(1);                                                                                                \
    }                                                                                                         \
  }                                                                                                           \
  while(0)


//
// terse description of the CUDA error codes (replacement of the error code number by its name)
//

static const char *cu_error_string(CUresult e)
{
  static char error_string[64];
# define CASE(error_code)  case error_code: return "" # error_code;
  switch((int)e)
  { // list of error codes extracted from cuda.h (TODO: /usr/local/cuda-10.2/targets/x86_64-linux/include/CL)
    default: sprintf(error_string,"unknown error code (%d)",(int)e); return(error_string);
    CASE(CUDA_SUCCESS                             );
    CASE(CUDA_ERROR_INVALID_VALUE                 );
    CASE(CUDA_ERROR_OUT_OF_MEMORY                 );
    CASE(CUDA_ERROR_NOT_INITIALIZED               );
    CASE(CUDA_ERROR_DEINITIALIZED                 );
    CASE(CUDA_ERROR_PROFILER_DISABLED             );
    CASE(CUDA_ERROR_PROFILER_NOT_INITIALIZED      );
    CASE(CUDA_ERROR_PROFILER_ALREADY_STARTED      );
    CASE(CUDA_ERROR_PROFILER_ALREADY_STOPPED      );
    CASE(CUDA_ERROR_NO_DEVICE                     );
    CASE(CUDA_ERROR_INVALID_DEVICE                );
    CASE(CUDA_ERROR_INVALID_IMAGE                 );
    CASE(CUDA_ERROR_INVALID_CONTEXT               );
    CASE(CUDA_ERROR_CONTEXT_ALREADY_CURRENT       );
    CASE(CUDA_ERROR_MAP_FAILED                    );
    CASE(CUDA_ERROR_UNMAP_FAILED                  );
    CASE(CUDA_ERROR_ARRAY_IS_MAPPED               );
    CASE(CUDA_ERROR_ALREADY_MAPPED                );
    CASE(CUDA_ERROR_NO_BINARY_FOR_GPU             );
    CASE(CUDA_ERROR_ALREADY_ACQUIRED              );
    CASE(CUDA_ERROR_NOT_MAPPED                    );
    CASE(CUDA_ERROR_NOT_MAPPED_AS_ARRAY           );
    CASE(CUDA_ERROR_NOT_MAPPED_AS_POINTER         );
    CASE(CUDA_ERROR_ECC_UNCORRECTABLE             );
    CASE(CUDA_ERROR_UNSUPPORTED_LIMIT             );
    CASE(CUDA_ERROR_CONTEXT_ALREADY_IN_USE        );
    CASE(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED       );
    CASE(CUDA_ERROR_INVALID_PTX                   );
    CASE(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT      );
    CASE(CUDA_ERROR_NVLINK_UNCORRECTABLE          );
    CASE(CUDA_ERROR_INVALID_SOURCE                );
    CASE(CUDA_ERROR_FILE_NOT_FOUND                );
    CASE(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
    CASE(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED     );
    CASE(CUDA_ERROR_OPERATING_SYSTEM              );
    CASE(CUDA_ERROR_INVALID_HANDLE                );
    CASE(CUDA_ERROR_NOT_FOUND                     );
    CASE(CUDA_ERROR_NOT_READY                     );
    CASE(CUDA_ERROR_ILLEGAL_ADDRESS               );
    CASE(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES       );
    CASE(CUDA_ERROR_LAUNCH_TIMEOUT                );
    CASE(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING );
    CASE(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED   );
    CASE(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED       );
    CASE(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE        );
    CASE(CUDA_ERROR_CONTEXT_IS_DESTROYED          );
    CASE(CUDA_ERROR_ASSERT                        );
    CASE(CUDA_ERROR_TOO_MANY_PEERS                );
    CASE(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
    CASE(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED    );
    CASE(CUDA_ERROR_HARDWARE_STACK_ERROR          );
    CASE(CUDA_ERROR_ILLEGAL_INSTRUCTION           );
    CASE(CUDA_ERROR_MISALIGNED_ADDRESS            );
    CASE(CUDA_ERROR_INVALID_ADDRESS_SPACE         );
    CASE(CUDA_ERROR_INVALID_PC                    );
    CASE(CUDA_ERROR_LAUNCH_FAILED                 );
    CASE(CUDA_ERROR_NOT_PERMITTED                 );
    CASE(CUDA_ERROR_NOT_SUPPORTED                 );
    CASE(CUDA_ERROR_UNKNOWN                       );
  };
# undef CASE
}


//
// synchonize the stream command buffer
//

static void synchronize_cuda(cuda_data_t *cd)
{
  CU_CALL( cuStreamSynchronize , (cd->cu_stream) );
}

//
// initialize the CUDA driver API interface
//
// load a single cubin file, with a single CUDA kernel
// allocate up to two storage areas both on the host and on the device
//

static void initialize_cuda(cuda_data_t *cd)
{
  //
  // initialize the driver API interface
  //
  CU_CALL( cuInit , (0) );
  //
  // open the CUDA device
  //
  CU_CALL( cuDeviceGet , (&cd->cu_device,cd->device_number) );
  //
  // get information about the CUDA device
  //
  CU_CALL( cuDeviceGetName , (cd->device_name,(int)sizeof(cd->device_name) - 1,cd->cu_device) );
  printf("initialize_cuda(): CUDA code running on a %s (device %d, CUDA %u.%u.%u)\n",cd->device_name,cd->device_number,CUDA_VERSION / 1000,(CUDA_VERSION / 10) % 100,CUDA_VERSION % 10);
  //
  // create a context
  //
  CU_CALL( cuCtxCreate , (&cd->cu_context, NULL, CU_CTX_SCHED_YIELD, cd->cu_device) );
  CU_CALL( cuCtxSetCacheConfig , (CU_FUNC_CACHE_PREFER_L1) );
  //
  // load precompiled modules
  //
  CU_CALL( cuModuleLoad , (&cd->cu_module,cd->cubin_file_name) );
  //
  // get the kernel function pointers
  //
  CU_CALL( cuModuleGetFunction, (&cd->cu_kernel,cd->cu_module,cd->kernel_name) );
  //
  // create a command stream (we could have used the default stream)
  //
  CU_CALL( cuStreamCreate, (&cd->cu_stream,CU_STREAM_NON_BLOCKING) );
  //
  // allocate host and device memory
  //
  for(int i = 0;i < 2;i++)
    if(cd->data_size[i] > 0u)
    {
      CU_CALL( cuMemAllocHost , ((void **)&cd->host_data[i]  ,(size_t)cd->data_size[i]) );
      CU_CALL( cuMemAlloc     ,          (&cd->device_data[i],(size_t)cd->data_size[i]) );
    }
    else
      cd->host_data[i] = NULL;
  //
  // catch any lingering errors
  //
  synchronize_cuda(cd);
}


//
// terminate the CUDA driver API interface
//

static void terminate_cuda(cuda_data_t *cd)
{
  CU_CALL( cuStreamDestroy, (cd->cu_stream) );
  for(int i = 0;i < 2;i++)
    if(cd->data_size[i] > 0u)
    {
      CU_CALL( cuMemFreeHost , (cd->host_data[i]) );
      CU_CALL( cuMemFree , (cd->device_data[i]) );
    }
  CU_CALL( cuModuleUnload , (cd->cu_module) );
  CU_CALL( cuCtxDestroy , (cd->cu_context) );
}


//
// copy data from the host to the device and from the device to the host
//

static void host_to_device_copy(cuda_data_t *cd,int idx)
{
  if(idx < 0 || idx > 1 || cd->data_size[idx] == 0u)
  {
    fprintf(stderr,"host_to_device_copy(): bad idx\n");
    exit(1);
  }
  CU_CALL( cuMemcpyHtoD , (cd->device_data[idx],(void *)cd->host_data[idx],(size_t)cd->data_size[idx]) );
  //synchronize_cuda(cd);
}

static void device_to_host_copy(cuda_data_t *cd,int idx)
{
  if(idx < 0 || idx > 1 || cd->data_size[idx] == 0u)
  {
    fprintf(stderr,"device_to_host_copy(): bad idx\n");
    exit(1);
  }
  CU_CALL( cuMemcpyDtoH , ((void *)cd->host_data[idx],cd->device_data[idx],(size_t)cd->data_size[idx]) );
  synchronize_cuda(cd);
}



//
// launch a CUDA kernel (with 0 bytes of shared memory and no extra options)
//

static void lauch_kernel(cuda_data_t *cd)
{
  if(cd->block_dim_x != (unsigned int)RECOMENDED_CUDA_BLOCK_SIZE)
    fprintf(stderr,"lauch_kernel(): block_dim_x should be equal to %d\n",RECOMENDED_CUDA_BLOCK_SIZE);
  CU_CALL( cuLaunchKernel , (cd->cu_kernel,cd->grid_dim_x,1u,1u,cd->block_dim_x,1u,1u,0u,cd->cu_stream,&cd->arg[0],NULL) );
  //synchronize_cuda(cd);
}

#endif
message.txt
11 KB