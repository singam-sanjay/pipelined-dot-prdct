#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<omp.h>

#include "../env_decl.h"

struct TID
{
  int x;
};

int num_threads_in_tree(int N)
{
  int sum = N;
  do
  {
    N = (N>>1) + (N&1);
    sum += N;
  }while(N>1);
  return sum;
}

int N, k;
TYPE *vec, *mat, *res;

#define FS_TYPE PRINTF_FORMAT_STRING /*Shortend MACRO*/

#define CLASS_LEAF (0b01)
#define CLASS_ROOT (0b10)
#define CLASS_INTM (0b11)
#define is_CLASS_LEAF (class_of_thread==CLASS_LEAF)
#define is_CLASS_ROOT (class_of_thread==CLASS_ROOT)
#define have_2nd_MEM_OPRND (class_of_thread&2)
#define save_to_SHRD_MEM (class_of_thread&1)
#define save_to_GLBL_MEM (is_CLASS_ROOT)
/* NEW STUFF */
#define is_ACTIVE (class_of_thread&3)
#define just_became_ACTIVE (iter==0)
#define IDLE_and_DONE (class_of_thread==0)
/*/NEW STUFF */

TYPE *sw_cache;

void pipeline_kernel( int num_of_threadz )
{
  /* Global OpenMP and other stuff for adapting the CUDA code */
  struct TID blockDim = {num_of_threadz};
  omp_set_dynamic(0);//To disable dynamic teams (http://stackoverflow.com/questions/11095309)
  omp_set_num_threads(num_of_threadz);
  sw_cache = (double*)malloc( sizeof(TYPE)*num_of_threadz );
  //_syncthreads(); is replaced by #pragma omp barrier using 'sed'
  /* End of OpenMP stuff */
  volatile int max_lvl;

  #pragma omp parallel
  {
    /* OpenMP Stuff for adapting the CUDA code */
    TID threadIdx;
    threadIdx.x = omp_get_thread_num();
    /* End of OpenMP stuff */
    TYPE *src1,*src2,op1,op2,*dest;
    int iter,c1,c2,lvl=0,wait;
    unsigned char class_of_thread = '\0';
    if( threadIdx.x<N )
    {
      src1 = mat;
      src2 = vec;
      dest = sw_cache;
      c1 = c2 = threadIdx.x;
      op2 = vec[c2]; //load the reference vector.
      class_of_thread = CLASS_LEAF;
    }
    else
    {
      int N_lvl = N, prev_lvl_1st, curr_lvl_1st = 0, nxt_lvl_1st = N;
      do
      {
        prev_lvl_1st = curr_lvl_1st;
        curr_lvl_1st = nxt_lvl_1st;
        N_lvl = (N_lvl/2) + (N_lvl%2); // N_lvl = ceil(N/2) = N/2 + N%2 = (N>>1) + N&1
        nxt_lvl_1st = curr_lvl_1st + N_lvl;
        ++lvl;
      }while( (curr_lvl_1st>threadIdx.x) || (threadIdx.x>=nxt_lvl_1st) );//!(curr_lvl_1st<=threadIdx.x && threadIdx.x<nxt_lvl_1st)
      c1 = prev_lvl_1st + ((threadIdx.x-curr_lvl_1st)*2);
      if( (c1+1)<nxt_lvl_1st )
      {
        c2 = c1+1; //using c2 as c1+1 since avoiding 1.extra addition including reg access.  and  2.repetetive addition
      }
      else
      {
        c2 = blockDim.x-1; //assuming the last thread's __shared__ spot is a 'source of zeros', sw_cache[blockDim.x-1]==0, since right child doesn't exist.
      }
      src1 = src2 = sw_cache;
      if( threadIdx.x == blockDim.x-1 )
      {
        dest = res;
        sw_cache[threadIdx.x] = lvl;
        class_of_thread = CLASS_ROOT;
      }
      else
      {
        dest = sw_cache;
        class_of_thread = CLASS_INTM;
      }
      class_of_thread <<= 2;//These threads are idle, initially
    }

    //printf("b4: threadIdx.x:%i lvl:%i max_lvl:%i wait:%i\n",threadIdx.x,lvl,max_lvl,wait);fflush(stdout);
    iter = -lvl;
    __syncthreads(); //waiting for max_lvl
    max_lvl = sw_cache[blockDim.x-1];
    __syncthreads(); //Waiting to for all to read max_lvl
    wait = max_lvl-lvl;
    if( threadIdx.x==blockDim.x-1)
    {
      sw_cache[threadIdx.x] = 0; //init 'source of zeros' only once to avoid 1.redundant writes  and  2.serialised writes to same memory bank
      //The 1st __syncthreads in while would ensure this write be visible to all threads
    }
    /*DEBUGSTUFF*///printf("aftr: ID:%i iter:%i lvl:%i max_lvl:%i wait:%i\n",threadIdx.x,iter,lvl,max_lvl,wait);fflush(stdout);
    /*DEBUGSTUFF*/TYPE op1_save;
    while(true)
    {
      __syncthreads();
      op1 = src1[c1];
      /*DEBUGSTUFF*/op1_save = op1;
      if( have_2nd_MEM_OPRND )
      {
        op2 = src2[c2];
        op1 += op2;
      }
      else if( is_CLASS_LEAF )
      {
        op1 -= op2;
        op1 *= op1;
        src1 += N;//Go to next vector
      }
      /*DEBUGSTUFF*//*
      printf("ID:%i iter:%i class:%i wait:%i op1:"FS_TYPE" op2:"FS_TYPE" res:"FS_TYPE"\n", threadIdx.x, iter, class_of_thread, wait, op1_save, op2, op1);fflush(stdout);
      __syncthreads();
      if( threadIdx.x==0 )
      {
        static char chumma[10];
        printf("Waiting...");
        scanf("%s",chumma);
      }*/
      /*/DEBUGSTUFF*/
      __syncthreads();
      if( save_to_SHRD_MEM )
      {
        dest[threadIdx.x] = op1;
      }
      else if( save_to_GLBL_MEM )
      {
        dest[iter] = op1;
      }

      ++iter;
      if( is_ACTIVE )//Changed control structure since active threads were going through all the conditions
      {
        if( iter<k )
        {
          continue;
        }
        else // iter==k
        {
          class_of_thread = 0;
          //Idle threads keep accessing memory, to prevent that direct then to some safe source (that doesn't cost much)
          src1 = sw_cache;
          // Now all Idle threads would be hitting the same location, so BROADCAST :)
          c1 = blockDim.x-1;
        }
      }
      else if( just_became_ACTIVE )
      {
        class_of_thread>>=2;
      }
      else if( IDLE_and_DONE )
      {
        if( !(wait) )
        {
          break;//This was return in the CUDA kernel, although can still be 'break' even in the CUDA code.
        }
        --wait; //tID==blockDim.x-1 doesn't have to wait at all
      }

    }
  }
  free(sw_cache);
}

void gen_and_print_input_data();
void print_res();

int main(int argc, char** argv)
{
  if( argc!=3 )
  {
    printf("./a.out N k\n");
    return EXIT_FAILURE;
  }
  N = atoi(argv[1]);
  k = atoi(argv[2]);
  vec = (TYPE*)malloc(sizeof(TYPE)*N);
  if(vec==NULL){printf("vec is NULL");return 1;}
  mat = (TYPE*)malloc(sizeof(TYPE)*N*k);
  if(vec==NULL){free(vec);printf("mat is NULL");return 1;}
  res = (TYPE*)malloc(sizeof(TYPE)*k);
  if(res==NULL){free(mat);free(vec);printf("res is NULL");return 1;}

  int num_of_threadz;
  num_of_threadz = num_threads_in_tree(N);

  gen_and_print_input_data();
  pipeline_kernel(num_of_threadz);
  print_res();
  free(vec);
  free(mat);
  free(res);
}

void gen_and_print_input_data()
{
  register int i1,i2;
  srand(time(NULL));
  printf("vec:");
  for( i1=0 ; i1<N ; i1++ )
  {
    vec[i1] = rand();
    printf(" "PRINTF_FORMAT_STRING,vec[i1]);
  }
  printf("\nmat:\n");
  int row_offset = 0;
  for( i1=0 ; i1<k ; ++i1 )
  {
    for( i2=0 ; i2<N ; ++i2 )
    {
      mat[row_offset+i2] = rand();
      printf(" "PRINTF_FORMAT_STRING,mat[row_offset+i2]);
    }
    row_offset += N;
    putchar('\n');
  }
}

void print_res()
{
  register int i1;
  for( i1=0 ; i1<k ; ++i1 )
  {
    printf(FS_TYPE"\n",res[i1]);
  }
  fflush(stdout);
}
