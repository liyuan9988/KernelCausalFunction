rm(list=ls())
library(causalweight)


################
# Application
################

#load the data and define covariates X (please adapt path accordingly)
load("C:\\JCdata.RData")
attach(JCdata)
x=cbind(female, age, race_white, race_black, race_hispanic,  hgrd, hgrdmissdum, educ_geddiploma, educ_hsdiploma, ntv_engl, marstat_divorced, marstat_separated, marstat_livetogunm, marstat_married, haschldY0, everwkd,  mwearn, hohhd0, peopleathome, peopleathomemissdum, nonres,    g10, g10missdum, g12, g12missdum, hgrd_mum, hgrd_mummissdum, hgrd_dad, hgrd_dadmissdum, work_dad_didnotwork, g2, g5, g7, welfare_child, welfare_childmissdum, h1_fair_poor, h2, h10, h10missdum,  h25, h25missdum, h29, h5, h5missdum, h7, h7missdum, i1, i10, e12, e12missdum, e16, e16missdum, e21, e24usd, e24usdmissdum, e30, e30missdum, e31, e32, e35, e35missdum, e37, e6_byphone, e8_recruitersoffice, e9ef)

# estimate direct and indirect treatment effects using the main specification (dropping obs. with d=0) 
dropped=c();results.de.treat=c();results.de.control=c(); results.ie.treat=c();results.ie.control=c(); results.de.treat.t=c();results.de.control.t=c(); results.ie.treat.t=c();results.ie.control.t=c();
d0=40; eval=seq(100, 2000, 100);
for (d1 in eval){
  temp=medweightcont(y=y[d>0],d=d[d>0],m=m[d>0],x=x[d>0,], d0=d0, d1=d1, ATET = FALSE, trim = 0.1, lognorm = TRUE, bw = NULL, boot = 999, cluster = NULL)
  dropped=c(dropped,temp$ntrimmed)
  results=temp$results
  results.de.treat=rbind(results.de.treat,cbind(results[1,2], results[1,2]-1.96*results[2,2],results[1,2]+1.96*results[2,2]))
  results.de.control=rbind(results.de.control,cbind(results[1,3], results[1,3]-1.96*results[2,3],results[1,3]+1.96*results[2,3]))
  results.ie.treat=rbind(results.ie.treat,cbind(results[1,4], results[1,4]-1.96*results[2,4],results[1,4]+1.96*results[2,4]))
  results.ie.control=rbind(results.ie.control,cbind(results[1,5], results[1,5]-1.96*results[2,5],results[1,5]+1.96*results[2,5]))
}

# average number of dropped observations
mean(dropped)

# plots of the effects
plot(x=eval, y=results.de.treat[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.04), xlab="hours in academic and/or vocational training", ylab="direct effect under treatment" )
lines(x=eval, y=results.de.treat[,2], type="l", lty=2)
lines(x=eval, y=results.de.treat[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.de.control[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.04), xlab="hours in academic and/or vocational training", ylab="direct effect under non-treatment" )
lines(x=eval, y=results.de.control[,2], type="l", lty=2)
lines(x=eval, y=results.de.control[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.ie.treat[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.04), xlab="hours in academic and/or vocational training", ylab="indirect effect under treatment" )
lines(x=eval, y=results.ie.treat[,2], type="l", lty=2)
lines(x=eval, y=results.ie.treat[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.ie.control[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.04), xlab="hours in academic and/or vocational training", ylab="indirect effect under non-treatment" )
lines(x=eval, y=results.ie.control[,2], type="l", lty=2)
lines(x=eval, y=results.ie.control[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)


# estimate direct and indirect treatment effects using all observations and setting d0=0 
dropped=c();results.de.treat=c();results.de.control=c(); results.ie.treat=c();results.ie.control=c(); results.de.treat.t=c();results.de.control.t=c(); results.ie.treat.t=c();results.ie.control.t=c();
d0=0; eval=seq(100, 2000, 100);
for (d1 in eval){
  temp=medweightcont(y=y,d=d,m=m,x=x, d0=d0, d1=d1, ATET = FALSE, trim = 0.1, lognorm = TRUE, bw = NULL, boot = 999, cluster = NULL)
  dropped=c(dropped,temp$ntrimmed)
  results=temp$results
  results.de.treat=rbind(results.de.treat,cbind(results[1,2], results[1,2]-1.96*results[2,2],results[1,2]+1.96*results[2,2]))
  results.de.control=rbind(results.de.control,cbind(results[1,3], results[1,3]-1.96*results[2,3],results[1,3]+1.96*results[2,3]))
  results.ie.treat=rbind(results.ie.treat,cbind(results[1,4], results[1,4]-1.96*results[2,4],results[1,4]+1.96*results[2,4]))
  results.ie.control=rbind(results.ie.control,cbind(results[1,5], results[1,5]-1.96*results[2,5],results[1,5]+1.96*results[2,5]))
}


# plots of the effects
plot(x=eval, y=results.de.treat[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.08), xlab="hours in academic and/or vocational training", ylab="direct effect under treatment" )
lines(x=eval, y=results.de.treat[,2], type="l", lty=2)
lines(x=eval, y=results.de.treat[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.de.control[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.08), xlab="hours in academic and/or vocational training", ylab="direct effect under non-treatment" )
lines(x=eval, y=results.de.control[,2], type="l", lty=2)
lines(x=eval, y=results.de.control[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.ie.treat[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.08), xlab="hours in academic and/or vocational training", ylab="indirect effect under treatment" )
lines(x=eval, y=results.ie.treat[,2], type="l", lty=2)
lines(x=eval, y=results.ie.treat[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)

plot(x=eval, y=results.ie.control[,1], type="b", lty=1, xlim=c(100, 2000), ylim=c(-0.14, 0.08), xlab="hours in academic and/or vocational training", ylab="indirect effect under non-treatment" )
lines(x=eval, y=results.ie.control[,2], type="l", lty=2)
lines(x=eval, y=results.ie.control[,3],type="l", lty=2)
lines(x=c(0,eval), y=rep(0,length(eval)+1),type="l", lty=3)