ddpath  <- "/Users/liyuanxu/PycharmProjects/KGF/code/data/job_corps_raw/"
setwd(ddpath)

load(file=paste (ddpath ,  "my_jcUNIFR.RData",  sep=""))
attach(jcUNIFR)
Y = M3
D = totalhrs_acadvocy1
X1 = data.frame(female=female,
               age=age,
               race_white=race_white,
               race_black=race_black,
               race_hispanic=race_hispanic,
               hgrd=hgrd,
               hgrdmissdum=hgrdmissdum,
               educ_geddiploma=educ_geddiploma,
               educ_hsdiploma=educ_hsdiploma,
               ntv_engl=ntv_engl,
               marstat_divorced=marstat_divorced,
               marstat_separated=marstat_separated,
               marstat_livetogunm=marstat_livetogunm,
               marstat_married=marstat_married,
               haschldY0=haschldY0,
               everwkd=everwkd,
               mwearn=mwearn,
               hohhd0=hohhd0,
               peopleathome=peopleathome,
               peopleathomemissdum=peopleathomemissdum,
               nonres=nonres,
               g10=g10,
               g10missdum=g10missdum,
               g12=g12,
               g12missdum=g12missdum,
               hgrd_mum=hgrd_mum,
               hgrd_mummissdum=hgrd_mummissdum,
               hgrd_dad=hgrd_dad,
               hgrd_dadmissdum=hgrd_dadmissdum,
               work_dad_didnotwork=work_dad_didnotwork,
               g2=g2,
               g5=g5,
               g7=g7,
               welfare_child=welfare_child,
               welfare_childmissdum=welfare_childmissdum,
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
               i10=i10,
               e12=e12,
               e12missdum=e12missdum,
               e16=e16,
               e16missdum=e16missdum,
               e21=e21,
               e24usd=e24usd,
               e24usdmissdum=e24usdmissdum,
               e30=e30,
               e30missdum=e30missdum,
               e31=e31,
               e32=e32,
               e35=e35,
               e35missdum=e35missdum,
               e37=e37,
               e6_byphone=e6_byphone,
               e8_recruitersoffice=e8_recruitersoffice,
               e9ef=e9ef)

D = D[!is.na(Y)]
X1 = X1[!is.na(Y),]
Y = Y[!is.na(Y)]

write.csv(D, "D.csv", row.names = F, col.names = F)
write.csv(X1, "X1.csv", row.names = F, col.names = F)
write.csv(Y, "Y.csv", row.names = F, col.names = F)