INCLUDE_DIRECTORIES(
    ../
)

ADD_LIBRARY(detection detection.cc)
ADD_LIBRARY(voc_tree voc_tree.cc)
ADD_LIBRARY(voc_tree_database voc_tree_database.cc)
ADD_LIBRARY(voc_tree_inv_file voc_tree_inv_file.cc)
