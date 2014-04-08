/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_LOOP_SORT_H_
#define MAVMAP_SRC_LOOP_SORT_H_


#include <stdio.h>
#include <memory.h>


class HeapSort {

public:

  inline void sortarray(unsigned int *A, int n) {
    int ex;
    buildmaxheap(A, n);
    for (int i = n; i >= 2; i--) {
      ex = A[0];
      A[0] = A[i-1];
      A[i-1] = ex;
      n--;
      maxheapify(A,1,n);

    }
  }


  inline void buildmaxheap(unsigned int *A, int n) {
    for (int i = n/2; i >= 1; i--)
      maxheapify(A, i, n);
  }


  inline void maxheapify(unsigned int *A, int i, int n) {
    int l = 2*i;
    int r = 2*i+1;
    int largest;
    int ex;
    if (l <= n && A[l-1] > A[i-1])
      largest = l;
    else largest = i;
    if (r <= n && A[r-1] > A[largest-1])
      largest = r;
    if (largest != i) {
      ex = A[largest-1];
      A[largest-1] = A[i-1];
      A[i-1] = ex;
      maxheapify(A,largest,n);
    }
  }


  inline void sortarray(float *A, int n, int *x) {
    int ex;
    float fex;
    buildmaxheap(A, n,x);
    for (int i = n; i >= 2; i--) {
      fex = A[0];
      A[0] = A[i-1];
      A[i-1] = fex;
      ex = x[0];
      x[0] = x[i-1];
      x[i-1] = ex;
      n--;
      maxheapify(A,1,n,x);

    }
  }


  inline void buildmaxheap(float *A, int n, int *x) {
    for (int i = n/2; i >= 1; i--)
      maxheapify(A, i, n,x);
  }


  inline void maxheapify(float *A, int i, int n, int *x) {
    int l = 2*i;
    int r = 2*i+1;
    int largest;
    int ex;
    float fex;
    if (l <= n && A[l-1] > A[i-1])
      largest = l;
    else largest = i;
    if (r <= n && A[r-1] > A[largest-1])
      largest = r;
    if (largest != i) {
      fex = A[largest-1];
      A[largest-1] = A[i-1];
      A[i-1] = fex;
      ex = x[largest-1];
      x[largest-1] = x[i-1];
      x[i-1] = ex;
      maxheapify(A,largest,n,x);
    }

  }


  inline void sortarray(unsigned int *A, int n, float *x, float *y) {
    int ex;
    float fex;
    buildmaxheap(A, n,x,y);
    for (int i = n; i >= 2; i--) {
      ex = A[0];
      A[0] = A[i-1];
      A[i-1] = ex;
      fex = x[0];
      x[0] = x[i-1];
      x[i-1] = fex;
      fex = y[0];
      y[0] = y[i-1];
      y[i-1] = fex;
      n--;
      maxheapify(A,1,n,x,y);

    }
  }


  inline void buildmaxheap(unsigned int *A, int n, float *x, float *y) {
    for (int i = n/2; i >= 1; i--)
      maxheapify(A, i, n,x,y);
  }


  inline void maxheapify(unsigned int *A, int i, int n, float *x, float *y) {
    int l = 2*i;
    int r = 2*i+1;
    int largest;
    int ex;
    float fex;
    if (l <= n && A[l-1] > A[i-1])
      largest = l;
    else largest = i;
    if (r <= n && A[r-1] > A[largest-1])
      largest = r;
    if (largest != i) {
      ex = A[largest-1];
      A[largest-1] = A[i-1];
      A[i-1] = ex;
      fex = x[largest-1];
      x[largest-1] = x[i-1];
      x[i-1] = fex;
      fex = y[largest-1];
      y[largest-1] = y[i-1];
      y[i-1] = fex;
      maxheapify(A,largest,n,x,y);
    }

  }


  inline void sortarray(unsigned int *A, int n, float *x, float *y, float *s) {
    int ex;
    float fex;
    buildmaxheap(A, n,x,y,s);
    for (int i = n; i >= 2; i--) {
      ex = A[0];
      A[0] = A[i-1];
      A[i-1] = ex;
      fex = x[0];
      x[0] = x[i-1];
      x[i-1] = fex;
      fex = y[0];
      y[0] = y[i-1];
      y[i-1] = fex;
      fex = s[0];
      s[0] = s[i-1];
      s[i-1] = fex;
      n--;
      maxheapify(A,1,n,x,y);

    }
  }


  inline void buildmaxheap(unsigned int *A, int n,
                           float *x, float *y, float *s) {
    for (int i = n/2; i >= 1; i--)
      maxheapify(A, i, n,x,y);
  }


  inline void maxheapify(unsigned int *A, int i, int n,
                         float *x, float *y, float *s) {
    int l = 2*i;
    int r = 2*i+1;
    int largest;
    int ex;
    float fex;
    if (l <= n && A[l-1] > A[i-1])
      largest = l;
    else largest = i;
    if (r <= n && A[r-1] > A[largest-1])
      largest = r;
    if (largest != i) {
      ex = A[largest-1];
      A[largest-1] = A[i-1];
      A[i-1] = ex;
      fex = x[largest-1];
      x[largest-1] = x[i-1];
      x[i-1] = fex;
      fex = y[largest-1];
      y[largest-1] = y[i-1];
      y[i-1] = fex;
      fex = s[largest-1];
      s[largest-1] = s[i-1];
      s[i-1] = fex;
      maxheapify(A,largest,n,x,y);
    }

  }

};


// removes duplicates from sorted list in place
inline void unique(unsigned int *A, unsigned int *n) {
  unsigned int last = 0;
  unsigned int current = 0;
  while (current < *n) {
    while (A[last] == A[current]) {  // move on
      current++;
    }
    last++;
    A[last] = A[current];
  }
  *n = last;
}


// removes duplicates from sorted list and x and y lists in place
inline void unique(unsigned int *A, unsigned int *n, float *x, float *y) {
  unsigned int last = 0;
  unsigned int current = 0;
  while (current < *n) {
    while (A[last] == A[current]) {  // move on
      current++;
    }
    last++;
    A[last] = A[current];
    x[last] = x[current];
    y[last] = y[current];
  }
  *n = last;
}

#endif // MAVMAP_SRC_LOOP_SORT_H_
