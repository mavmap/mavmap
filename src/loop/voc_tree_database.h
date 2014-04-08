/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_LOOP_VOC_TREE_DATABASE_H_
#define MAVMAP_SRC_LOOP_VOC_TREE_DATABASE_H_

#include <math.h>

#include "voc_tree_inv_file.h"
#include "sort.h"


#define DATABASE_RANSAC 500


class VocTreeDatabase: public VocTreeInvFile {

public:

  VocTreeDatabase(void);
  ~VocTreeDatabase(void);

  // Class initialization and memory allocation
  // Param: nrdocs ... (IN) maximum number of documents in database
  //        nrvisualwords ... (IN) range of vw's quantization of voctree
  //        maxvw ....... (IN) max number of vw per document
  int init(int nrdocs, int nrvisualwords, int maxvw);

  // Inserts a document into database. "docname" is a label for the document.
  // On query the label gets returned.
  // Param: vw ... (IN) visual words of the document
  //        nr ... (IN) length of visual word vector
  //        docname ... (IN) label for document
  int insertdoc(unsigned int *vw, int nr, int docname);
  int insertdoc(unsigned int *vw, int nr, int docname, float *x, float *y);

  int match(int docpos, unsigned int *vw, int nr, float *x, float *y,
            float *x1, float *y1, float *x2, float *y2);
  int getforwarddata(int docpos, float *x, float *y, float *s);
  int getdocvw(int docpos, unsigned int *vw);
  void normalize(void);

  void clean(void);

private:

  int m_init;
  unsigned int *m_vw;
  int *m_nrvw;
  float *m_x;
  float *m_y;
  float *m_s;
  int *m_blockpt;
  int m_blockcount;
  float **m_svdU;
  float *m_svdW;
  float **m_svdV;
  float **m_Harr;
  float **m_ptarr;
  float *m_gscore;

};

#endif // MAVMAP_SRC_LOOP_VOC_TREE_DATABASE_H_
