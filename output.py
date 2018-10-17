import parameters
import numpy as np

def output_xenos(outfilename,x0,y0,dose):
    repetitions = dose / (parameters.dwell_time*parameters.current)
    repetitions[np.where(repetitions < 1)] = 0
    repetitions = np.array(np.round(repetitions), dtype=np.int)

    Outputfile = open(outfilename,'w')

    #Outputfile.write('D line_test, 11000, 11000, 5, 5' + '\n')
    Outputfile.write('D line_test' + '\n')
    Outputfile.write('I 1' + '\n')
    Outputfile.write('C '+str(int(parameters.dwell_time*1e9)) + '\n')
    Outputfile.write("FSIZE 15 micrometer" + '\n')
    Outputfile.write("UNIT 1 micrometer" + '\n')

    print(repetitions)
    for j in range(len(x0)):
        if repetitions[j] >= 1:
            Outputfile.write('RDOT '+str(x0[j]) + ', ' + str(y0[j]) + ', ' + str((repetitions[j])) + '\n')
    Outputfile.write('END' + '\n')
    Outputfile.write('\n')
    Outputfile.write('\n')

    Outputfile.close()
