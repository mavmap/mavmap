/**
 * Copyright (C) 2013
 *
 *   Johannes L. Schönberger <johannes.schoenberger (at) tum.de>
 *   Friedrich Fraundorfer <friedrich.fraundorfer (at) tum.de>
 *
 */

#include "voc_tree_inv_file.h"


using namespace std;


VocTreeInvFile::VocTreeInvFile(void) {
  m_init = 0;
}


VocTreeInvFile::~VocTreeInvFile(void) {
  if (m_init == 1) {
    clean();
    m_init = 0;
  }
}


int VocTreeInvFile::init(int nrdocs, int nrvisualwords, int maxvw) {

  m_maxdocs = nrdocs;
  m_nrvisualwords = nrvisualwords;
  m_maxvw = maxvw;
  m_maxblocks = fmax(5*nrdocs*maxvw/INV_BLOCKSIZE,100000);
  m_blocknr = 0;
  m_docnr = 0;
  m_blockspace = new invblock[m_maxblocks];
  m_invnr = new unsigned int[m_nrvisualwords];
  m_inv = new invblock*[m_nrvisualwords];
  m_idf = new float[m_nrvisualwords];

  for (int i = 0; i < m_nrvisualwords; ++i) {
    m_invnr[i] = 0;
    m_inv[i] = NULL;
  }
  for (int j = 0; j < m_maxblocks; ++j) {
    m_blockspace[j].next = NULL;
  }

  m_score = new float[m_maxdocs+1];
  m_scoreadd = new float[m_maxdocs];
  m_scorepos = new int[m_maxdocs*m_maxvw];
  m_docnames = new int[m_maxdocs];
  m_docsums = new float[m_maxdocs];
  m_docnormalizer = new float[m_maxdocs];
  m_docindex = new int[m_nrvisualwords];
  memset(m_docindex, 0, sizeof(int)*m_nrvisualwords);

  for (int j = 0; j < m_nrvisualwords; ++j) {
    m_idf[j] = log((float)m_maxdocs);
  }

  m_init  = 1;
  return 0;
}


void VocTreeInvFile::clean(void) {
  if (m_init == 1) {
    delete [] m_blockspace;
    delete [] m_invnr;
    delete [] m_inv;
    delete [] m_idf;
    delete [] m_score;
    delete [] m_scoreadd;
    delete [] m_scorepos;
    delete [] m_docnames;
    delete [] m_docsums;
    delete [] m_docnormalizer;
    delete [] m_docindex;
    m_init = 0;
  }
}



int VocTreeInvFile::insert(unsigned int *vw, int nr, int docname) {
  if (m_docnr == m_maxdocs) return -1;
  int pos;
  invblock* pt;
  m_docnames[m_docnr] = docname;
  // compute m_docsums = |di| sum of weights for scoring
  m_docsums[m_docnr] = 0;
  m_docnormalizer[m_docnr] = 0;
  for (int i = 0; i < nr; ++i) {
    m_docindex[vw[i]]++;
  }
  for (int i = 0; i < nr; ++i) {
    if (m_docindex[vw[i]] > 0) {
      m_docsums[m_docnr] += ((float)m_docindex[vw[i]]*m_idf[vw[i]]
                             *(float)m_docindex[vw[i]]*m_idf[vw[i]]);
      m_docindex[vw[i]] = 0;
    }
  }
  m_docnormalizer[m_docnr] = (float)1./sqrt(m_docsums[m_docnr]);

  for (int i = 0; i < nr; ++i) {
    // compute relative position in block
    pos = m_invnr[vw[i]] % INV_BLOCKSIZE;
    // find last block
    pt = m_inv[vw[i]];
    if (pt == NULL) {  // attach first block
      pt = m_inv[vw[i]] = &m_blockspace[m_blocknr];
      m_blocknr++;
    }
    while (pt->next != NULL) {
      pt = pt->next;
    }
    pt->data[pos] = m_docnr;
    m_invnr[vw[i]]++;
    if (pos+1 == INV_BLOCKSIZE) {  // attach new block
      pt->next = &m_blockspace[m_blocknr];
      m_blocknr++;
      if (m_blocknr > m_maxblocks)
        cout << "OUT OF MEMORY: VocTreeInvFile::insert" << endl;
    }
  }
  m_docnr++;
  return m_docnr-1;
}


void VocTreeInvFile::computeidf(void) {
  for (int j = 0; j < m_nrvisualwords; ++j) {
    m_idf[j] = 0;
  }

  invblock* pt;
  int *idfcount = new int[m_docnr];
  int block;
  //int maxcount = 0;
  for (int i = 0; i < m_nrvisualwords; ++i) {
    m_idf[i] = 0;
    if (m_invnr[i] > 0) {
      memset(idfcount, 0, sizeof(int)*m_docnr);

      pt = m_inv[i];
      block = 0;
      while (pt != NULL) {
        for (int j = 0; j < fmin((int)m_invnr[i]-block*INV_BLOCKSIZE,
                                 (int)INV_BLOCKSIZE); ++j) {
          idfcount[pt->data[j]] = 1;
        }
        pt = pt->next;
        block++;
      }

      for (int j = 0; j < m_docnr; ++j) {
        m_idf[i] += idfcount[j];
      }
      //if(m_idf[i]>maxcount)maxcount = m_idf[i];
      m_idf[i] =  log(float(m_docnr)/m_idf[i]);
    }
  }
  delete [] idfcount;

}


int VocTreeInvFile::query(unsigned int *vw, int nr) {
  // best match = score 0
  // worst match = score 2

  int scoreposnr;
  int vwi;
  float bestscore = 2;
  int bestindex = -1;
  invblock* pt;
  int block;

  float querysum = 0;
  float normalizer = 0;
  for (int i = 0; i < nr; ++i) {
    m_docindex[vw[i]]++;
  }
  for (int i = 0; i < nr; ++i) {
    if (m_docindex[vw[i]] > 0) {
      querysum += ((float)m_docindex[vw[i]]*m_idf[vw[i]]
                   *(float)m_docindex[vw[i]]*m_idf[vw[i]]);
      m_docindex[vw[i]] = 0;
    }
  }
  normalizer = (float)1./sqrt(querysum);


  memset(m_scoreadd, 0, sizeof(float)*m_maxdocs);
  memset(m_score, 0, sizeof(float)*m_maxdocs);



  for (int j = 0; j < nr; ++j) {
    vwi = vw[j];
    scoreposnr = 0;

    pt = m_inv[vwi];
    block = 0;
    while (pt != NULL) {
      for (int k = 0; k < fmin((int)m_invnr[vwi]-block*INV_BLOCKSIZE,
                              (int)INV_BLOCKSIZE); ++k) {
        m_scoreadd[pt->data[k]] = m_idf[vwi];
        m_scorepos[scoreposnr] = pt->data[k];
        scoreposnr++;
      }
      pt = pt->next;
      block++;
    }
    for (int k = 0; k < scoreposnr; ++k) {
      m_score[m_scorepos[k]] += (normalizer*m_idf[vwi]
                                 *m_docnormalizer[m_scorepos[k]]
                                 *m_scoreadd[m_scorepos[k]]);
    }
    for (int k = 0; k < scoreposnr; ++k) {
      m_scoreadd[m_scorepos[k]] = 0;
    }
  }

  for (int i = 0; i < m_docnr; ++i)
    m_score[i] = (float)(2.-2.*fmin(m_score[i],(float)1.));
    // due to numerical inaccuracies (m_score is computed from many small
    // numbers) m_score could be slightly higher than 1, which would result in
    // a score smaller than 0. This will be captured here by the min()
    // function, which limits m_score to 1.

  for (int i = 0; i < m_docnr; ++i)
    if (m_score[i] < bestscore) {
      bestscore = m_score[i];
      bestindex = i;
    }

  return m_docnames[bestindex];
}


int VocTreeInvFile::querytopn(unsigned int *vw, int nr, int n, int *docnames,
                              float *scores, int sort) {
  // best match = score 0
  // worst match = score 2

  int scoreposnr;
  int vwi;
  invblock* pt;
  int block;

  // compute querysum = |qi| sum of weights for scoring
  float querysum = 0;
  float normalizer = 0;
  for (int i = 0; i < nr; ++i) {
    m_docindex[vw[i]]++;
  }
  for (int i = 0; i < nr; ++i) {
    if (m_docindex[vw[i]] > 0) {
      querysum += ((float)m_docindex[vw[i]]*m_idf[vw[i]]
                   *(float)m_docindex[vw[i]]*m_idf[vw[i]]);
      m_docindex[vw[i]] = 0;
    }
  }
  normalizer = (float)1./sqrt(querysum);

  memset(m_scoreadd, 0, sizeof(float)*m_maxdocs);
  memset(m_score, 0, sizeof(float)*m_maxdocs);

  for (int j = 0; j < nr; ++j) {
    vwi = vw[j];
    scoreposnr = 0;

    pt = m_inv[vwi];
    block = 0;
    while (pt != NULL) {
      for (int k = 0; k < fmin((int)m_invnr[vwi]-block*INV_BLOCKSIZE,
                               (int)INV_BLOCKSIZE); ++k) {
        m_scoreadd[pt->data[k]] = m_idf[vwi];
        m_scorepos[scoreposnr] = pt->data[k];
        scoreposnr++;
      }
      pt = pt->next;
      block++;
    }
    for (int k = 0; k < scoreposnr; ++k) {
      m_score[m_scorepos[k]] += (normalizer*m_idf[vwi]
                                 *m_docnormalizer[m_scorepos[k]]
                                 *m_scoreadd[m_scorepos[k]]);
    }
    for (int k = 0; k < scoreposnr; ++k) {
      m_scoreadd[m_scorepos[k]] = 0;
    }
  }

  for (int i = 0; i < m_docnr; ++i)
    m_score[i] = (float)(2.-2.*fmin(m_score[i],(float)1.));
    // due to numerical inaccuracies (m_score is computed from many small
    // numbers) m_score could be slightly higher than 1, which would result in
    // a score smaller than 0. This will be captured here by the min()
    // function, which limits m_score to 1.

  if (sort == 1) {
    // get best n scores
    for (int i = 0; i < n; ++i) {
      scores[i] = 20;
      docnames[i] = -1;
    }

    int docpos = 0;
    for (int jj = 0; jj < n; ++jj) {
      for (int k = 0; k < m_docnr; ++k) {
        if (m_score[k] < scores[jj]) {
          scores[jj] = m_score[k];
          docnames[jj] = m_docnames[k];
          docpos = k;
        }
      }
      m_score[docpos] = 20;
    }
  } else {
    for (int i = 0; i < m_docnr; ++i)
      scores[i] = m_score[i];
  }

  return docnames[0];
}


void VocTreeInvFile::saveidf(char *fname) {
  FILE *fidf;
  fidf = fopen(fname, "wb");
  fwrite(m_idf, sizeof(float), m_nrvisualwords, fidf);
  fclose(fidf);
}


void VocTreeInvFile::loadidf(char *fname) {
  FILE *fidf;
  fidf = fopen(fname, "rb");
  fread(m_idf, sizeof(float), m_nrvisualwords, fidf);
  fclose(fidf);
}
