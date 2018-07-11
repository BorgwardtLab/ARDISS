import math
from scipy.stats.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error

def elapsed_time(time_in_s):
    # function returning a string of elapsed time in minutes/hours
    if time_in_s < 60:
        return "{} s".format(time_in_s)
    elif time_in_s < 3600:
        return "{} m {} s".format(int(math.floor(time_in_s/60)), time_in_s % 60)
    else:
        return "{} h {} m {} s".format(int(math.floor(time_in_s/3600)), int(math.floor((time_in_s % 3600)/60)), (time_in_s%3600%60))

def verboseprint(string, verbose=True):
    if verbose:
        print(string)

def compareZsc(ifile, mfile, suffix = ""):
    """Usage:
    -i (required) specify imputed file name
    -m (required) specify original masked file name
    -s (optional) specify suffix of data directories (empty by default)"""

    # Build dictionary of typed SNPs that were initially masked
    f = open(mfile, 'r')
    f.readline()
    snps_dict = dict()
    for line in f:
        line = line[:-1]
        cols = line.split()
        snp_id = cols[0]
        snps_dict[snp_id] = cols[4]
    f.close()

    # find imputed SNPs corresponding to filtered SNPs
    f = open(ifile, 'r')
    f.readline()
    snps = []
    sq = 0
    i = 0
    for line in f:
        line = line[:-1]
        cols = line.split()
        snp_id = cols[0]
        if snp_id in snps_dict:
            i = i+1
            squared = (float(snps_dict[snp_id]) - float(cols[4])) ** 2
            sq +=squared
            snps.append((snp_id+'\t'+snps_dict[snp_id]+'\t'+cols[4]+'\t'+str(squared)))

    f.close()

    print("The sum of squared error is : "+str(sq)+"\nThe mean squared error is : "+str(sq/i))

    # write new file
    f = open(suffix+".comparison", 'w')
    f.write("SNP_id\ttyped_Z-score\timputed_Z-score\tsquared_error\n")
    for n in snps:
        f.write(n+"\n")

    f.close()


def correlation(ifile, genome_region=""):
    """Usage:
    -i (required) specify input file name"""

    # The input file has the format: SNP_id, typed_Z-score, imputed Z-score
    # Extract the two vectors of Z-scores
    f = open(ifile, 'r')
    typedZ = []
    imputedZ = []
    f.readline()
    for line in f:
        cols = line[:-1].split()
        if not math.isnan(float(cols[2])):
            typedZ.append(float(cols[1]))
            imputedZ.append(float(cols[2]))
    f.close()

    m = 0
    t = 0
    for i in range(len(imputedZ)):
        if (typedZ[i]>0 and imputedZ[i]<0) or (typedZ[i]<0 and imputedZ[i]>0):
            m = m+1
        t = t+1
    # Get chromosome number if single chr:
    if genome_region == "":
        chr_num = ifile.split("/")[-1].split(".")[0][3:]
        corrstr = "The Pearson coefficient R for imputed scores on chromosome " + chr_num +" is : " + str(pearsonr(typedZ, imputedZ))
    else:
        corrstr = "The Pearson coefficient R for imputed scores on " + genome_region + " is : " + str(
            pearsonr(typedZ, imputedZ))
    print(corrstr)
    # print(str(m)+" over " + str(t) + " ("+str(float(m)/float(t)*100)+"%) imputed points have opposite directions")
    return corrstr, pearsonr(typedZ,imputedZ)

def r2(ifile, genome_region=""):
    """Usage:
    -i (required) specify input file name"""

    # The input file has the format: SNP_id, typed_Z-score, imputed Z-score
    # Extract the two vectors of Z-scores
    f = open(ifile, 'r')
    typedZ = []
    imputedZ = []
    f.readline()
    for line in f:
        cols = line[:-1].split()
        if not math.isnan(float(cols[2])):
            typedZ.append(float(cols[1]))
            imputedZ.append(float(cols[2]))
    f.close()

    m = 0
    t = 0
    for i in range(len(imputedZ)):
        if (typedZ[i]>0 and imputedZ[i]<0) or (typedZ[i]<0 and imputedZ[i]>0):
            m = m+1
        t = t+1
    # Get chromosome number if single chr:
    if genome_region == "":
        chr_num = ifile.split("/")[-1].split(".")[0][3:]
        corrstr = "The R2 score for imputed scores on chromosome " + chr_num +" is : " + str(r2_score(typedZ, imputedZ))
    else:
        corrstr = "The R2 score for imputed scores on " + genome_region + " is : " + str(
            r2_score(typedZ, imputedZ))
    print(corrstr)
    print(str(m)+" over " + str(t) + " ("+str(float(m)/float(t)*100)+"%) imputed points have opposite directions")
    return corrstr, r2_score(typedZ, imputedZ)

def mse(ifile, genome_region=""):
    """Usage:
    Return Mean Squared error"""
    # The input file has the format: SNP_id, typed_Z-score, imputed Z-score
    # Extract the two vectors of Z-scores
    f = open(ifile, 'r')
    typedZ = []
    imputedZ = []
    f.readline()
    for line in f:
        cols = line[:-1].split()
        if not math.isnan(float(cols[2])):
            typedZ.append(float(cols[1]))
            imputedZ.append(float(cols[2]))
    f.close()

    err = mean_squared_error(typedZ, imputedZ)
    # Get chromosome number if single chr:
    if genome_region == "":
        chr_num = ifile.split("/")[-1].split(".")[0][3:]
        corrstr = "The Mean squared error for imputed scores on chromosome " + chr_num +" is : " + str(err)
    else:
        corrstr = "The Mean squared error for imputed scores on " + genome_region + " is : " + str(
            err)
    print(corrstr)
    return corrstr, err
