/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "voc_tree.h"


using namespace std;


VocTree::VocTree(void) {
  m_init = 0;
}


VocTree::~VocTree(void) {
  if (m_init == 1) {
    clean();
    m_init = 0;
  }
}


int VocTree::init(const char *fname, int truncate) {
  // load voctree
  FILE *fin;
  fin = fopen(fname, "rb");
  if(fin == 0) {
    return 0;
  }
  int ret = 0;
  ret = fread(&m_visualwords, sizeof(m_visualwords), 1, fin);
  ret = fread(&m_levels, sizeof(m_levels), 1, fin);
  ret = fread(&m_splits, sizeof(m_splits), 1, fin);
  ret = fread(&m_nrcenters, sizeof(m_nrcenters), 1, fin);
  if(m_levels > 10 || m_splits > 100000) {
    m_visualwords = 0;
    m_levels = 0;
    m_splits = 0;
    m_nrcenters = 0;
    fclose(fin);
    return 0;
  }
  if (truncate) {
    m_levels--;
    int  nc = m_nrcenters - m_visualwords;
    m_voc = new uint8_t[nc*128];
    ret = fread(m_voc, sizeof(uint8_t), nc*128, fin);
    ret = fseek(fin, m_visualwords*128, SEEK_CUR);
    m_cellinfo = new uint8_t[nc];
    ret = fread(m_cellinfo, sizeof(uint8_t), nc, fin);
    m_nrcenters = nc;
    m_visualwords /= m_splits;
  }
  else {
    m_voc = new uint8_t[m_nrcenters*128];
    ret = fread(m_voc, sizeof(uint8_t), m_nrcenters*128, fin);
    m_cellinfo = new uint8_t[m_nrcenters];
    ret = fread(m_cellinfo, sizeof(uint8_t) ,m_nrcenters, fin);
  }
  fclose(fin);

  m_vwtable = new uint32_t[m_nrcenters];

  uint32_t abspos = 0;
  // create vwtable
  for (uint32_t level = 0; level < m_levels; level++) {
    uint32_t step = (uint32_t)(m_visualwords/pow((double)m_splits,
                                                 (double)level+1));
    for (uint32_t i = 0; i < pow((double)m_splits, (double)level+1); ++i) {
      m_vwtable[abspos] = step*i;
      abspos++;
    }
  }

  m_init = 1;
  return 1;
}


void VocTree::clean(void) {
  if (m_init == 1) {
    delete [] m_voc;
    delete [] m_cellinfo;
    delete [] m_vwtable;
    m_init = 0;
  }
}


void VocTree::quantize(uint32_t *vwi, const uint8_t *desc) const {
  int32_t m_centerpos = 0;
  int32_t m_previndex = 0;
  int32_t m_offset = -1;
  for (uint32_t level = 0; level < m_levels; level++) {

    m_offset = (m_offset + 1)*m_splits;
    m_centerpos = (m_centerpos+m_previndex)*m_splits;

    //mindist = 256*5000; // L1
    double m_mindist = 256*256*5000; // L2
    int32_t m_minindex = 0;

    uint8_t * voc = m_voc + (m_centerpos+m_offset)*128;
    for (uint32_t i = 0; i < m_splits; ++i) {
      double m_dist = 0;
      const uint8_t * ps = desc;
      for (int k = 0; k < 128; ++k, ps++, voc++) {
        int diff = ps[0]-voc[0];
        //dist += abs(diff));    //L1
        m_dist += (diff*diff);    //L2
      }
      if (m_dist < m_mindist) {
        m_mindist = m_dist;
        m_minindex = i;
      }
    }

    m_previndex = m_minindex;

    if (m_cellinfo[m_centerpos+m_offset] > 0) {
      *vwi = m_vwtable[m_centerpos+m_offset+m_previndex];
      return;
    }
  }
  *vwi = m_vwtable[m_centerpos+m_offset+m_previndex];
}


int VocTree::calcnrcenters(int splits, int levels) {
  return (int)(pow((double)splits, (double)levels+1)-1)/(splits-1) - 1;
}
