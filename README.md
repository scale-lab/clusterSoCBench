# clusterSoCBench
clusterSoCBench is a collection of benchmarks ported to ARM that can be used to evaluate the CPU+GPGPU performance of an ARM-based cluster.

For more details please visit our publication:

R. Azimi, T. Fox and S. Reda, "Understanding the Role of GPGPU-accelerated SoC-based ARM Clusters", to appear in IEEE Cluster 2017.


# Build instructions
Copy the source code to all nodes and compile each benchmark using make.
For HPL, we used OPENBLAS. For message passing library:, we used OpenMPI.   

More information about the individual benchmarks can be found here:

TeaLeaf: [https://github.com/UK-MAC/TeaLeaf]

CloverLeaf: [https://github.com/UK-MAC/CloverLeaf]

Jacobi: [https://github.com/parallel-forall/code-samples/tree/master/posts/cuda-aware-mpi-example/src]
  

# Contacts
If you use any of our ideas, please cite our paper.  Our contact information is:  
reza_azimi at brown dot edu, tyler_fox at brown dot edu or sherif_reda at brown dot edu

