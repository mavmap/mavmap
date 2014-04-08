/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "voc_tree_database.h"


using namespace std;


VocTreeDatabase::VocTreeDatabase(void) {
  m_init = 0;
}


VocTreeDatabase::~VocTreeDatabase(void) {
  if (m_init == 1) {
    clean();
    m_init = 0;
  }
}


int VocTreeDatabase::init(int nrdocs, int nrvisualwords, int maxvw) {

  VocTreeInvFile::init(nrdocs, nrvisualwords, maxvw);
  m_vw = new unsigned int[nrdocs*maxvw];
  m_nrvw = new int[nrdocs];
  m_x = new float[nrdocs*maxvw];
  m_y = new float[nrdocs*maxvw];
  m_blockpt = new int[nrdocs];
  m_blockcount = 0;
  m_svdU = new float*[9];
  for (int i = 0; i < 9; ++i)
    m_svdU[i] = new float[9];
  m_svdW = new float[9];
  m_svdV = new float*[9];
  for (int i = 0; i < 9; ++i)
    m_svdV[i] = new float[9];

  m_Harr = new float*[DATABASE_RANSAC];
  for (int i = 0; i < DATABASE_RANSAC; ++i)
    m_Harr[i] = new float[9];
  m_ptarr = new float*[DATABASE_RANSAC];
  for (int i = 0; i < DATABASE_RANSAC; ++i)
    m_ptarr[i] = new float[16];
  m_gscore = new float[DATABASE_RANSAC];

  return 0;
}

void VocTreeDatabase::clean(void) {
  if (m_init == 1) {
    VocTreeInvFile::clean();
    delete [] m_vw;
    delete [] m_nrvw;
    delete [] m_x;
    delete [] m_y;
    delete [] m_blockpt;
    for (int i = 0; i < 9; ++i) {
      delete [] m_svdU[i];
      delete [] m_svdV[i];
    }
    delete [] m_svdU;
    delete [] m_svdW;
    delete [] m_svdV;
    for (int i = 0; i < DATABASE_RANSAC; ++i) {
      delete [] m_Harr[i];
      delete [] m_ptarr[i];
    }
    delete [] m_Harr;
    delete [] m_ptarr;
    delete [] m_gscore;

    m_init = 0;
  }
}


int VocTreeDatabase::insertdoc(unsigned int *vw, int nr, int docname) {
  if (m_docnr == m_maxdocs) return -1;
  // add to forward file
  m_nrvw[m_docnr] = nr;
  m_blockpt[m_docnr] = m_blockcount;
  memcpy(&m_vw[m_blockcount], vw, sizeof(unsigned int)*nr);
  m_blockcount += nr;

  return VocTreeInvFile::insert(vw, nr, docname);
}


int VocTreeDatabase::insertdoc(unsigned int *vw, int nr, int docname,
                               float *x, float *y) {
  if (m_docnr == m_maxdocs) return -1;
  // add to forward file
  m_nrvw[m_docnr] = nr;
  m_blockpt[m_docnr] = m_blockcount;
  memcpy(&m_vw[m_blockcount], vw, sizeof(unsigned int)*nr);
  memcpy(&m_x[m_blockcount], x, sizeof(float)*nr);
  memcpy(&m_y[m_blockcount], y, sizeof(float)*nr);
  m_blockcount += nr;

  return VocTreeInvFile::insert(vw, nr, docname);
}


int VocTreeDatabase::match(int docpos, unsigned int *vw, int nr,
                           float *x, float *y, float *x1, float *y1,
                           float *x2, float *y2) {
  // VocTreedatabase is considered to be left image
  // VocTreedatabase points are returned in x1,y1
  // query points are copied to x2,y2

  int ivwdb;
  int ivw;
  int matchnr = 0;

  ivw = 0;
  ivwdb = m_blockpt[docpos];

  while (ivw < nr && ivwdb-m_blockpt[docpos] < m_nrvw[docpos]) {
    while (vw[ivw] < m_vw[ivwdb]) {
      ivw++;
    }
    while (vw[ivw] > m_vw[ivwdb]) {
      ivwdb++;
    }
    if (vw[ivw] == m_vw[ivwdb]) {
      // matching word found; sorted, unique vw list required!
      x1[matchnr] = m_x[ivwdb];
      y1[matchnr] = m_y[ivwdb];
      x2[matchnr] = x[ivw];
      y2[matchnr] = y[ivw];
      matchnr++;
      ivw++;
      ivwdb++;
    }
  }


  return matchnr;
}


int VocTreeDatabase::getforwarddata(int docpos, float *x, float *y, float *s) {
  int nr = m_nrvw[docpos];
  int m_blockcount = m_blockpt[docpos];
  memcpy(x, &m_x[m_blockcount], sizeof(float)*nr);
  memcpy(y, &m_y[m_blockcount], sizeof(float)*nr);
  memcpy(s, &m_s[m_blockcount], sizeof(float)*nr);
  return nr;
}


int VocTreeDatabase::getdocvw(int docpos, unsigned int *vw) {
  int nr = m_nrvw[docpos];
  int m_blockcount = m_blockpt[docpos];
  memcpy(vw, &m_vw[m_blockcount], sizeof(unsigned int)*nr);
  return nr;
}


void VocTreeDatabase::normalize(void) {

  for (int j = 0; j < m_docnr; ++j) {
    int ivwdb = m_blockpt[j];
    int nr = m_nrvw[j];

    m_docsums[j] = 0;
    m_docnormalizer[j] = 0;
    for (int i = 0; i < nr; ++i) {
      m_docindex[m_vw[ivwdb+i]]++;
    }
    for (int i = 0; i < nr; ++i) {
      int ivwdb_i = ivwdb+i;
      if (m_docindex[m_vw[ivwdb_i]] > 0) {
        m_docsums[j] += (float)m_docindex[m_vw[ivwdb_i]]*m_idf[m_vw[ivwdb_i]]
                        *(float)m_docindex[m_vw[ivwdb_i]]*m_idf[m_vw[ivwdb_i]];
        m_docindex[m_vw[ivwdb_i]] = 0;
      }
    }
    m_docnormalizer[j] = (float)1./sqrt(m_docsums[j]);
  }
}
