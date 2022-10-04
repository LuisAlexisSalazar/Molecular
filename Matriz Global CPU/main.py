from utils import *
from Meddleman import Matrix
import CPU_threads_scores
from copy import copy
import time
import numpy as np


def saveValueMatrix(valueMatrix, n_cores):
    # np.savetxt('Arreglo de valores.txt', self.values_matrix, fmt='%.0f')
    np.savetxt("output/output" + str(n_cores) + ".txt", valueMatrix, fmt='%.0f', header="Matrix de Valores:")


# A T G G C C A T T
# A T C C A A T T T T
# tcaagcgtta gagaagtcat tatgtgataa aaaaattcaa cttggtatca acttaactaagggtcttggt gctggtgctt tgcctgatgt tggtaaaggt gcagcagaag aatcaattgatgaaattatg gagcatataa aagatagcca tatgctcttt atcacagcag ggatgggtggtggtactgga acaggtgctg caccggtaat tgcaaaagca gccagagaag caagagcggtagttaaagat aaaggagcaa aagaaaaaaa gatactgact gttggagttg taactaagccgttcggtttt gaaggtgtgc gacgtatgcg cattgcagag cttggacttg aagagttgcaaaaatacgta gatacactta ttgtcattcc caatcaaaat ttatttagaa ttgctaacgagaaaactaca tttgctgacg catttcaact cgccgataat gttctgcata ttggcataagaggagtaact gatttgatga tcatgccagg actgattaat cttgattttg ctgatatagaaacagtaatg agtgagatgg gtaaagcaat gattggtact ggagaggcag aaggagaagatagggcaatt agcgctgcag aggctgcgat atctaatcca ttacttgata atgtatcaatgaaaggtgca caaggaatat taattaatat tactggtggt ggagatatga ctctatttgaagttgatgct gcagccaata gagtgcgtga agaagtcgat gaaaatgcaa atataatatttggtgccact tttgatcagg cgatggaagg aagagttaga gtttctattc ttgcaactggcattgatagc tgtaacgaca attcatctgt taatcaaaac aagatcccag cagaggaaaaaaattttaaa tggccttata atcaagttcc tatatcagaa acaaaagaat atgcttcaactgagcaaaca aacgaaagag ttaagtgggg cagcaatgtt tatgatatac cagcttatctaagaagaaaa aaataatgca attttggcta ctcaagtcgg
# attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatctgttctctaaa cgaactttaa aatctgtgtg gctgtcactc ggctgcatgc ttagtgcactcacgcagtat aattaataac taattactgt cgttgacagg acacgagtaa ctcgtctatcttctgcaggc tgcttacggt ttcgtccgtg ttgcagccga tcatcagcac atctaggtttcgtccgggtg tgaccgaaag gtaagatgga gagccttgtc cctggtttca acgagaaaacacacgtccaa ctcagtttgc ctgttttaca ggttcgcgac gtgctcgtac gtggctttggagactccgtg gaggaggtct tatcagaggc acgtcaacat cttaaagatg gcacttgtggcttagtagaa gttgaaaaag gcgttttgcc tcaacttgaa cagccctatg tgttcatcaaacgttcggat gctcgaactg cacctcatgg tcatgttatg gttgagctgg tagcagaactcgaaggcatt cagtacggtc gtagtggtga gacacttggt gtccttgtcc ctcatgtgggcgaaatacca gtggcttacc gcaaggttct tcttcgtaag aacggtaata aaggagctggtggccatagt tacggcgccg atctaaagtc atttgactta ggcgacgagc ttggcactgatccttatgaa gattttcaag aaaactggaa cactaaacat agcagtggtg ttacccgtgaactcatgcgt gagcttaacg gaggggcata cactcgctat gtcgataaca acttctgtggccctgatggc taccctcttg agtgcattaa agaccttcta gcacgtgctg gtaaagcttcatgcactttg tccgaacaac tggactttat tgacactaag aggggtgtat actgctgccgtgaacatgag catgaaattg cttggtacac ggaacgttct gaaaagagct atgaattgcagacacctttt gaaattaaat tggcaaagaa atttgacacc ttcaatgggg aatgtccaaa
# ?Link del simulador del algoritmo de Meddleman
# https://bioboot.github.io/bimm143_W20/class-material/nw/
if __name__ == '__main__':
    # s1, s2 = readInput()
    # MatrixMeddleman = Matrix(s1, s2, debug=False)
    # MatrixMeddleman.fun(s1, s2)
    # matrix = MatrixMeddleman.values_matrix
    # my_list = []
    #
    # print("Matrix \n\n", matrix)
    # # print(len(matrix))
    # for f in range(1, len(matrix[0])):
    #     s = []
    #     for k in range(matrix.shape[1] - f):
    #         s.append(matrix[len(matrix) - k - 1, f + k])
    #         if len(s) == matrix.shape[1] - (matrix.shape[1]-matrix.shape[0]):
    #             break
    #     my_list.append(s)
    # print("Debajo de la diagonal")
    # for i in my_list:
    #     print(i)

    # print([n.tolist() for n in diags])

    n_cores_available = get_amount_cores()
    s1, s2 = readInput()
    MatrixMeddleman = Matrix(s1, s2, debug=False)
    copy_Matrix_Meddleman = copy(MatrixMeddleman.values_matrix)
    # copy_Matrix_Meddleman = MatrixMeddleman.values_matrix
    # *Serial
    marker = time.time()
    MatrixMeddleman.fun(s1, s2)
    print(MatrixMeddleman.values_matrix)
    print("Serial spent", time.time() - marker)
    MatrixMeddleman.saveValueMatrix()

    # *Threads
    CPU_threads_scores.values_matrix = copy_Matrix_Meddleman
    CPU_threads_scores.s1, CPU_threads_scores.s2 = s1, s2
    CPU_threads_scores.n_cores_available = n_cores_available
    n_cores_to_run = 4
    CPU_threads_scores.call_Cores_calculate_score(n_cores_to_run)
    saveValueMatrix(CPU_threads_scores.values_matrix, n_cores_to_run)

    # ! PoolProcess -> Muy malo por tener diferente scope
    # CPU_process_scores.values_matrix = copy(MatrixMeddleman.values_matrix)
    # CPU_process_scores.s1, CPU_process_scores.s2 = s1, s2
    # CPU_process_scores.matrix_coordinates = MatrixMeddleman.matrix_coordinates
    # marker = time.time()
    # call_Cores_calculate_score()
    # print("Multithreading 2 spent", time.time() - marker)
    #
    # MatrixMeddleman.fun(s1, s2)
    # print(MatrixMeddleman.values_matrix)
    # print(MatrixMeddleman.values_matrix.dtype)
    # MatrixMeddleman.alignments(s1, s2)
    # MatrixMeddleman.saveTXT()
