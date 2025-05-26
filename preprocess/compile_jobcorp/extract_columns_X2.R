ddpath  <- "/Users/liyuanxu/PycharmProjects/KGF/code/data/job_corps_raw/"
setwd(ddpath)

load(file=paste (ddpath ,  "my_jcUNIFR.RData",  sep=""))
attach(jcUNIFR)
Y = M3
D = totalhrs_acadvocy1

Y = M3
X1 = data.frame(age=age,
                hgrd=hgrd,
                hgrdmissdum=hgrdmissdum,
                educ_geddiploma=educ_geddiploma,
                educ_hsdiploma=educ_hsdiploma,
                marstat_divorced=marstat_divorced,
                marstat_separated=marstat_separated,
                marstat_livetogunm=marstat_livetogunm,
                marstat_married=marstat_married,
                haschldY0=haschldY0,
                mwearn=mwearn,
                peopleathome=peopleathome,
                peopleathomemissdum=peopleathomemissdum,
                nonres=nonres,
                g2=g2,
                g5=g5,
                g7=g7,
                h1_fair_poor=h1_fair_poor,
                h2=h2,
                h10=h10,
                h10missdum=h10missdum,
                h25=h25,
                h25missdum=h25missdum,
                h29=h29,
                h5=h5,
                h5missdum=h5missdum,
                h7=h7,
                h7missdum=h7missdum,
                i1=i1,
                i10=i10)


X1 = X1[!is.na(Y),]
write.csv(X1, "X2.csv", row.names = F, col.names = F)


