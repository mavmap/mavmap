/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

/*

  // Sample code-snipet:

  // load voctree
  VocTree voctree;
  voctree.init("voc1000000dog.bin");

  // quantize SIFT vectors into visual words (vw's)
  for (int i = 0; i < nr; ++i) {
    ...
    // quantizes a single SIFT vector into 1 visual word
    voctree.quantize(&vw, &sift);
    ...
  }

  // create inverted file
  ff_invfile inv;
  inv.init(nr, voctree.nrvisualwords(), 2000);

  // add images to database
  for (int i = 0; i < nr; ++i) {
    ...
    // adds a document vector (consisting of vwnr visual words) to database
    // under the denoted label
    inv.insert(&vw, vwnr, label);
    ...
  }

  // compute idf weighting for database
  inv.computeidf();

  // query image, returns the label of the 4 closest matches in the database
  inv.querytopn(&vw, vwnr, 4, qdocnames, qscores);

*/


#ifndef MAVMAP_SRC_LOOP_VOC_TREE_INV_FILE_H_
#define MAVMAP_SRC_LOOP_VOC_TREE_INV_FILE_H_

#define INV_BLOCKSIZE 10


#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>


class VocTreeInvFile {

  struct invblock {
    invblock *next;
    unsigned int data[INV_BLOCKSIZE];
  };

public:

  VocTreeInvFile(void);
  virtual ~VocTreeInvFile(void);

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
  int insert(unsigned int *vw, int nr, int docname);

  // Querying the database. Finds the closest database entry to the given
  // document and returns its label
  // Param: vw ... (IN) visual words of the query document
  //        nr ... (IN) length of visual word vector
  int query(unsigned int *vw, int nr);

  // Querying the database. Finds the n-closest database entries to the given
  // document and returns their labels and scores.
  // A vector for holding the labels and scores has to be provided.
  // Param: vw ... (IN) visual words of the query document
  //        nr ... (IN) length of visual word vector
  //        sort ..(IN) specifies if docnames are sorted according to scores
  //                    (sort == 1 ... sorted)
  int querytopn(unsigned int *vw, int nr, int n, int *docnames,
                float *scores, int sort=1);


  // Computes IDF weights from current database.
  void computeidf(void);

  // Stores IDF weights to file.
  void saveidf(char *fname);
  // Loads IDF weights from file.
  void loadidf(char *fname);

  void clean(void);

  inline int isvalid(void){return m_init;}

  unsigned int *m_invnr;

protected:

  int m_init;
  invblock* m_blockspace;
  invblock** m_inv;
  int m_nrvisualwords;
  int m_maxdocs;
  int m_maxvw;
  int m_blocknr;
  int m_docnr;
  int m_maxblocks;
  float* m_idf;
  float *m_score;
  float *m_scoreadd;
  int *m_scorepos;
  int *m_docnames;
  float *m_docsums;
  float *m_docnormalizer;
  int *m_docindex;

};

#endif // MAVMAP_SRC_LOOP_VOC_TREE_INV_FILE_H_
