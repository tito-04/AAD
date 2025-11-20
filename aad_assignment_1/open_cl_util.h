//
// Tomás Oliveira e Silva,  October 2024
//
// Arquiteturas de Alto Desempenho 2024/2025
//
// useful common stuff related to OpenCL
//
//

#ifndef OPEN_CL_UTIL
#define OPEN_CL_UTIL

//
// if necessary, include <CL/cl.hh>
//

#ifndef CL_SUCCESS
# define CL_USE_DEPRECATED_OPENCL_1_2_APIS  // for the clCreateCommandQueue function
# include <CL/cl.h>
#endif



//
// CL_CALL --- macro that should be used to call an OpenCL function and to test its return value
//
// it should be used to test the return value of calls such as
//   e = clGetPlatformIDs(1,&platform_id[0],&num_platforms);
//
// in this case, f_name is clGetPlatformIDs and args is (1,&platform_id[0],&num_platforms)
//

#define CL_CALL(f_name,args)                                                                                 \
  do                                                                                                         \
  {                                                                                                          \
    cl_int e = f_name args;                                                                                  \
    if(e != CL_SUCCESS)                                                                                      \
    { /* the call failed, terminate the program */                                                           \
      fprintf(stderr,"" # f_name "() returned %s (file %s,line %d)\n",cl_error_string(e),__FILE__,__LINE__); \
      exit(1);                                                                                               \
    }                                                                                                        \
  }                                                                                                          \
  while(0)


//
// CL_CALL_ALT --- another macro that should be used to call an OpenCL function and to test its return value
//
// it should be used the test the error code value of calls such as (error code returned via a pointer)
//   context = clCreateContext(NULL,1,&device_id[0],NULL,NULL,&e);
//
// in this case, f_name is context = clCreateContext and args is (NULL,1,&device_id[0],NULL,NULL,&e)
//

#define CL_CALL_ALT(f_name,args)                                                                              \
  do                                                                                                          \
  {                                                                                                           \
    cl_int e;                                                                                                 \
    f_name args;                                                                                              \
    if(e != CL_SUCCESS)                                                                                       \
    { /* the call failed, terminate the program */                                                            \
      fprintf(stderr,"" # f_name "() returned %s (file %s, line %d)\n",cl_error_string(e),__FILE__,__LINE__); \
      exit(1);                                                                                                \
    }                                                                                                         \
  }                                                                                                           \
  while(0)


//
// "user-friendly" description of the OpenCL error codes (replacement of the error code number by its name)
//

static const char *cl_error_string(cl_int e)
{
  static char error_string[64];
# define CASE(error_code)  case error_code: return "" # error_code;
  switch((int)e)
  { // list of error codes extracted from /usr/local/cuda-10.2/targets/x86_64-linux/include/CL/cl.h
    default: sprintf(error_string,"unknown error code (%d)",(int)e); return(error_string);
    CASE(CL_SUCCESS                                  );
    CASE(CL_DEVICE_NOT_FOUND                         );
    CASE(CL_DEVICE_NOT_AVAILABLE                     );
    CASE(CL_COMPILER_NOT_AVAILABLE                   );
    CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE            );
    CASE(CL_OUT_OF_RESOURCES                         );
    CASE(CL_OUT_OF_HOST_MEMORY                       );
    CASE(CL_PROFILING_INFO_NOT_AVAILABLE             );
    CASE(CL_MEM_COPY_OVERLAP                         );
    CASE(CL_IMAGE_FORMAT_MISMATCH                    );
    CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED               );
    CASE(CL_BUILD_PROGRAM_FAILURE                    );
    CASE(CL_MAP_FAILURE                              );
    CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET             );
    CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    CASE(CL_COMPILE_PROGRAM_FAILURE                  );
    CASE(CL_LINKER_NOT_AVAILABLE                     );
    CASE(CL_LINK_PROGRAM_FAILURE                     );
    CASE(CL_DEVICE_PARTITION_FAILED                  );
    CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE            );
    CASE(CL_INVALID_VALUE                            );
    CASE(CL_INVALID_DEVICE_TYPE                      );
    CASE(CL_INVALID_PLATFORM                         );
    CASE(CL_INVALID_DEVICE                           );
    CASE(CL_INVALID_CONTEXT                          );
    CASE(CL_INVALID_QUEUE_PROPERTIES                 );
    CASE(CL_INVALID_COMMAND_QUEUE                    );
    CASE(CL_INVALID_HOST_PTR                         );
    CASE(CL_INVALID_MEM_OBJECT                       );
    CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          );
    CASE(CL_INVALID_IMAGE_SIZE                       );
    CASE(CL_INVALID_SAMPLER                          );
    CASE(CL_INVALID_BINARY                           );
    CASE(CL_INVALID_BUILD_OPTIONS                    );
    CASE(CL_INVALID_PROGRAM                          );
    CASE(CL_INVALID_PROGRAM_EXECUTABLE               );
    CASE(CL_INVALID_KERNEL_NAME                      );
    CASE(CL_INVALID_KERNEL_DEFINITION                );
    CASE(CL_INVALID_KERNEL                           );
    CASE(CL_INVALID_ARG_INDEX                        );
    CASE(CL_INVALID_ARG_VALUE                        );
    CASE(CL_INVALID_ARG_SIZE                         );
    CASE(CL_INVALID_KERNEL_ARGS                      );
    CASE(CL_INVALID_WORK_DIMENSION                   );
    CASE(CL_INVALID_WORK_GROUP_SIZE                  );
    CASE(CL_INVALID_WORK_ITEM_SIZE                   );
    CASE(CL_INVALID_GLOBAL_OFFSET                    );
    CASE(CL_INVALID_EVENT_WAIT_LIST                  );
    CASE(CL_INVALID_EVENT                            );
    CASE(CL_INVALID_OPERATION                        );
    CASE(CL_INVALID_GL_OBJECT                        );
    CASE(CL_INVALID_BUFFER_SIZE                      );
    CASE(CL_INVALID_MIP_LEVEL                        );
    CASE(CL_INVALID_GLOBAL_WORK_SIZE                 );
    CASE(CL_INVALID_PROPERTY                         );
    CASE(CL_INVALID_IMAGE_DESCRIPTOR                 );
    CASE(CL_INVALID_COMPILER_OPTIONS                 );
    CASE(CL_INVALID_LINKER_OPTIONS                   );
    CASE(CL_INVALID_DEVICE_PARTITION_COUNT           );
  };
# undef CASE
}

#endif
