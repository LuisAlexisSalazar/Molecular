from utils import *
from Meddleman import Matrix

# ?Link del simulador del algoritmo de Meddleman (Global)
# https://bioboot.github.io/bimm143_W20/class-material/nw/
# ?Link del simulador del algoritmo (Local)
# https://www.ebi.ac.uk/Tools/psa/emboss_water/
# Gap Penalty:2  Mismatch score:2 y Match score:1
# https://observablehq.com/@manzt/smith-waterman-algorithm

# tcaagcgtta gagaagtcat tatgtgataa aaaaattcaa cttggtatca acttaactaa gggtcttggt gctggtgctt tgcctgatgt tggtaaaggt gcagcagaag aatcaattga
# attaaaggtt tataccttcc caggtaacaa accaaccaac tttcgatctc ttgtagatct gttctctaaa cgaactttaa aatctgtgtg gctgtcactc ggctgcatgc ttagtgcact
if __name__ == '__main__':
    s1, s2 = readInput()
    MatrixMeddleman = Matrix(s1, s2, debug=False, plot=False)
    MatrixMeddleman.fun(s1, s2)
    MatrixMeddleman.alignments(s1, s2)
    MatrixMeddleman.saveTXT(substring_same=True)
