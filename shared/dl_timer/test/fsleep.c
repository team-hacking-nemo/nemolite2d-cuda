#include <unistd.h>

int fsleep(int *nsecs){

  return sleep(*nsecs);
}
