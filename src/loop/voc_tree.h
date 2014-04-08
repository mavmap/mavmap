/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#ifndef MAVMAP_SRC_LOOP_VOC_TREE_H_
#define MAVMAP_SRC_LOOP_VOC_TREE_H_


#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>


class VocTree {
public:

  VocTree(void);
  ~VocTree(void);

  // Class initialization, memory allocation and voctree read from file
  // Param: fname ... (IN) path to voctree
  int init(const char *fname, int truncate = 0);

  // Memory deallocation
  void clean(void);

  // Quantize a 128 dimensional SIFT vector using the voctree.
  // Distances are computed in L2 norm.
  // Param: vwi ... (OUT) visual word for the SIFT vector
  //        sift... (IN) input SIFT vector 128 chars normalized to 0..255
  void quantize(uint32_t *vwi, const uint8_t *desc) const;
  // Computes the number of treenodes (clustercenters) for given split and
  // level.
  int calcnrcenters(int splits, int levels);

  // Returns number of levels for currently loaded voctree.
  inline int nrlevels(void) const { return m_levels; };
  // Returns number of splits for currently loaded voctree.
  inline int nrsplits(void) const { return m_splits; };
  // Returns number of centers for currently loaded voctree.
  inline int nrcenters(void) const { return m_nrcenters; };
  // Returns number of vw's for current loaded voctree.
  inline int nrvisualwords(void) const { return m_visualwords; };

  inline int isvalid(void) const { return m_init; }

private:

  int32_t m_init;
  uint8_t *m_voc;
  uint8_t *m_cellinfo;
  uint32_t *m_vwtable;
  uint32_t m_levels;
  uint32_t m_splits;
  uint32_t m_nrcenters;
  uint32_t m_visualwords;

};

#endif // MAVMAP_SRC_LOOP_VOC_TREE_H_
