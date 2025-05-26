# ----------------------------------------------------------------------------
#                                                       
#
#          CHAIR OF APPLIED ECONOMETRICS-EVALUATION OF PUBLIC POLICIES
#                        DEPARTMENT OF ECONOMICS 
#                   University of Fribourg (Switzerland)
#
#  AUTHOR:
#
#     Layal Pipoz
#
#  EMAIL:
#
#     layal.pipoz@unifr.ch
#
#  PROJECT:
#
#     MEDIATION ANALYSIS PROJECT WITH CONTINUOUS TREATMENT
#
#  DATE: 
#
#     August 2016
#
#  BUT:
#
#    Construction of the Dataset
#
#  DONNEES:
#
#     Type           : Panel (numeric)
#
#     N. obs.        : 15386
#  
#     File           : job_corps_dataset.rdata
#
#     Titre          : job01.r
#
#     Source         : Job Corps, USA
#
#     Description    : Mediation analysis with a continuous treatment variable
#
#     Variables      : 520 variables. See the excel file Variable_used_JC.
#
#
#-----------------------------------------------------------------------------

# Initialisation
# --------------

# Clear the workspace
rm(list=ls())

# Pathway to the data files

ddpath  <- "/Users/liyuanxu/PycharmProjects/KGF/code/data/job_corps_raw/"

#  Pathway to the work files

wdpath  <- "/Users/liyuanxu/PycharmProjects/KGF/code/data/job_corps_raw/"

# set the work and data files

setwd(ddpath) 

# Load the data

load(file=paste (ddpath ,  "my_job_corps_dataset.rdata",  sep=""))

# Libraries

library(plyr)# revalue

# Functions 
#----------

indicator<-function(condition) ifelse(condition,1,0)  #for the construction of dummies

 
# Missing dummies function
missdum<- function(x){
  Y<- data.frame(matrix(,nrow(x), ncol(x)))
  
  for(i in 1:nrow(x)){
    for(j in 1:ncol(x)){
     if(is.na(x[i,j])){
       Y[i,j]<- 1} else{Y[i,j]<-0}
}}
return(Y)}


# Period computation function
period<-function(x,y){
days <- numeric()
weeks<- numeric()

x <- as.Date(strptime(x, "%d/%m/%Y"))
y  <- as.Date(strptime(y, "%d/%m/%Y"))
days <- as.numeric(y - x)    
weeks<- as.numeric(days/7)

return(data.frame(days,weeks))}

# Construction of the dataset 

data <-job_corps_dataset

# Attach the data

attach(data)

#-----------------------------------------------------------------------------
# 1. Selection of the variables
#-----------------------------------------------------------------------------
                                                                                                                
#------------------------------------------------------------------------------
# 8-digit unique case identifier
#------------------------------------------------------------------------------

id <- mprid

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Continuous treatment
#------------------------------------------------------------------------------ 
#------------------------------------------------------------------------------ 

#-------------------------------------------------------------------------------
# 1. Construction of the total hours spent in academic programmes for any period during the 12 months following the randomization date
#-------------------------------------------------------------------------------
                              
# Treatment of the 97 values of the stop dates (97=still in programme)
#----------------------------------------------------------------------
rd_wk <- as.Date("1994-11-09") 
rd_wk_ind  <- rd_wk + (rand_wk-1)*7

tmp <- as.POSIXlt(rd_wk_ind)
tmp$year <- tmp$year+1
int_date <- as.Date(tmp)

tmp2<-as.POSIXlt(int_date)
   


# Treatment of the months which equal to 97 values
for(i in which(f_b21ma==97)){f_b21ma[i]<-tmp2$mon[i]+1 
                             f_b21ya[i]<-tmp2$year[i]
                             f_b21da[i]<-tmp2$mday[i]} #97 values   
                             
for(i in which(f_b21mb==97)){f_b21mb[i]<-tmp2$mon[i]+1 
                             f_b21yb[i]<-tmp2$year[i]                  
                             f_b21db[i]<-tmp2$mday[i]} #97 values  

for(i in which(f_b21mc==97)){f_b21mc[i]<-tmp2$mon[i]+1 
                             f_b21yc[i]<-tmp2$year[i]
                             f_b21dc[i]<-tmp2$mday[i]} #97 values  


for(i in which(f_b21md==97)){f_b21md[i]<-tmp2$mon[i]+1              
                             f_b21yd[i]<-tmp2$year[i]
                             f_b21dd[i]<-tmp2$mday[i]} #97 values  
  
                                                                  
                                                                                                  


# Treatment of the "does not know" values
f_b20da[f_b20da>=98]<-NA
f_b20ma[f_b20ma>=98]<-NA
f_b20ya[f_b20ya>=98]<-NA   

f_b21da[f_b21da>=98]<-NA      
f_b21ma[f_b21ma>=98]<-NA  
f_b21ya[f_b21ya>=98]<-NA         

                  
f_b20db[f_b20db>=98]<-NA
f_b20mb[f_b20mb>=98]<-NA
f_b20yb[f_b20yb>=98]<-NA   
      
f_b21db[f_b21db>=98]<-NA    
f_b21mb[f_b21mb>=98]<-NA  
f_b21yb[f_b21yb>=98]<-NA   

f_b20dc[f_b20dc>=98]<-NA
f_b20mc[f_b20mc>=98]<-NA
f_b20yc[f_b20yc>=98]<-NA   

f_b21dc[f_b21dc>=98]<-NA    
f_b21mc[f_b21mc>=98]<-NA  
f_b21yc[f_b21yc>=98]<-NA    

f_b20dd[f_b20dd>=98]<-NA
f_b20md[f_b20md>=98]<-NA
f_b20yd[f_b20yd>=98]<-NA   

f_b21dd[f_b21dd>=98]<-NA    
f_b21md[f_b21md>=98]<-NA  
f_b21yd[f_b21yd>=98]<-NA     
                        
#-------------------------------------------------------------------------------
# Computation of the number of days and weeks between the begin and the end of each attended academic program 
#
# Remarks:
# f_20ij=starting i(= day, month, year) of the respective academic programme (j=a,...,d), , j=a corresponds to acad1, j=b to acad2,... j=d to acad4
# f_21ij=stopping i(= day, month, year) of the respective academic programme (j=a,...,d)), , j=a corresponds to acad1, j=b to acad2,... j=g to acad4
# Function period computes the number of days and week  (see rows 91-101) from f_20ij and f_21ij
#-------------------------------------------------------------------------------
                                                                                                                                                                     
acad1_start<- paste(f_b20da,f_b20ma,f_b20ya, sep="/")
acad1_stop <- paste(f_b21da,f_b21ma,f_b21ya, sep="/")                                                                                                                                                               
acad1<-period(acad1_start,acad1_stop)
colnames(acad1)<- c("days_acad1", "weeks_acad1")

acad2_start<- paste(f_b20db,f_b20mb,f_b20yb, sep="/")
acad2_stop <- paste(f_b21db,f_b21mb,f_b21yb, sep="/")
acad2<-period(acad2_start,acad2_stop)  
colnames(acad2)<- c("days_acad2", "weeks_acad2")

acad3_start<- paste(f_b20dc,f_b20mc,f_b20yc, sep="/")
acad3_stop <- paste(f_b21dc,f_b21mc,f_b21yc, sep="/")
acad3<-period(acad3_start,acad3_stop)
colnames(acad3)<- c("days_acad3", "weeks_acad3") 

acad4_start<- paste(f_b20dd,f_b20md,f_b20yd, sep="/")
acad4_stop <- paste(f_b21dd,f_b21md,f_b21yd, sep="/")
acad4  <-period(acad4_start,acad4_stop)
colnames(acad4)<- c("days_acad4", "weeks_acad4")  

                                              
# Measurement errors treated as NA
#-------------------------------
# Negative values (due to programmes which began later than the 1st year after random assignement)
acad1[c(which(acad1$days_acad1<0)),]<-NA
acad2[c(which(acad2$days_acad2<0)),]<-NA
acad3[c(which(acad3$days_acad3<0)),]<-NA
acad4[c(which(acad4$days_acad4<0)),]<-NA

# Removal of the values equal to more than 365 days
acad1[c(which(acad1$days_acad1>365)),]<-NA
acad2[c(which(acad2$days_acad2>365)),]<-NA
acad3[c(which(acad3$days_acad3>365)),]<-NA
acad4[c(which(acad4$days_acad4>365)),]<-NA


#-------------------------------------------------------------------------------
# Computation of the total hours spent in any academics programmes during the 1st year after random assignment

# Variables: 
# f_b26j = number of weeks where the individual did not attend academics 1,2,3,4 (1 corresponds to j=a, 2 to j=b, 3 to j=c, 4 to j=d)
# f_b27j = number of days per week in academics 1,2,3,4 (j=a,b,c,d)
# f_b28j = number of hours per days in academics 1,2,3,4 (j=a,b,c,d)
#-------------------------------------------------------------------------------

# Acad 1
#---------

#Treatment of the 98 and 8 values
f_b26a[f_b26a>=98]<-NA
f_b26a[is.na(f_b26a)]<-0

f_b27a[f_b27a>=8]<-NA
f_b28a[f_b28a>=98]<-NA

A_hrs <- f_b27a*f_b28a # hours per week in acad 1
hrsacad1 <- A_hrs*(acad1$weeks_acad1-f_b26a) #total hours in academics 1 in the period where the individual took academics 1

                   
# Measurement errors treated as NA
# Negative values
hrsacad1[hrsacad1<0]<-NA
     
# Acad 2
#---------

#Treatment of the 98 and 8 values
f_b26b[f_b26b>=98]<-NA
f_b26b[is.na(f_b26b)]<-0

f_b27b[f_b27b>=8]<-NA
f_b28b[f_b28b>=98]<-NA

B_hrs <- f_b27b*f_b28b  # hours per week in acad 2
hrsacad2 <- B_hrs*(acad2$weeks_acad2-f_b26b)  #total hours in academics 2 in the period where the individual took academics 2
    
# Acad 3
#---------

#Treatment of the 98 and 8 values
f_b26c[f_b26c>=98]<-NA
f_b26c[is.na(f_b26c)]<-0

f_b27c[f_b27c>=8]<-NA
f_b28c[f_b28c>=98]<-NA

C_hrs <- f_b27c*f_b28c   # hours per week in acad 3
hrsacad3 <- C_hrs*(acad3$weeks_acad3-f_b26c)  #total hours in academics 3 in the period where the individual took academics 3

# Acad 4
#---------

#Treatment of the 98 and 8 values
f_b26d[f_b26d>=98]<-NA
f_b26d[is.na(f_b26d)]<-0

f_b27d[f_b27d>=8]<-NA
f_b28d[f_b28d>=98]<-NA

D_hrs <- f_b27d*f_b28d   # hours per week in acad 4  
hrsacad4 <- D_hrs*(acad4$weeks_acad4-f_b26d)  #total hours in academics 4 in the period where the individual took academics 4
                                                      
# Dataframe with of all hours spent in academics (periods 1-4)
#-------------------------------------------------------------
hrs_acad <- data.frame(id,
                       hrsacad1, 
                       hrsacad2,
                       hrsacad3,
                       hrsacad4)
                                 
# Total hours spent in any academics programmes during the 1st year
#-----------------------------------------------------------------  
S <- function(x) sum(x, na.rm=TRUE)                        
hrs_acad$total_hrs_acady1  <- apply(hrs_acad[,-1], 1, S)

#-------------------------------------------------------------------------------
# 2. Construction of the total hours spent in vocational trainings for any period
#-------------------------------------------------------------------------------
                              
# Treatment of the 97 values of the stop dates
#--------------------------------------------
   
# Treatment of the months
for(i in which(f_b39ma==97)){f_b39ma[i]<-tmp2$mon[i]+1 
                             f_b39ya[i]<-tmp2$year[i]
                             f_b39da[i]<-tmp2$mday[i]} #97 values   
                             
for(i in which(f_b39mb==97)){f_b39mb[i]<-tmp2$mon[i]+1 
                             f_b39yb[i]<-tmp2$year[i]                  
                             f_b39db[i]<-tmp2$mday[i]} #97 values  

for(i in which(f_b39mc==97)){f_b39mc[i]<-tmp2$mon[i]+1 
                             f_b39yc[i]<-tmp2$year[i]
                             f_b39dc[i]<-tmp2$mday[i]} #97 values  


for(i in which(f_b39md==97)){f_b39md[i]<-tmp2$mon[i]+1              
                             f_b39yd[i]<-tmp2$year[i]
                             f_b39dd[i]<-tmp2$mday[i]} #97 values  
  
for(i in which(f_b39me==97)){f_b39me[i]<-tmp2$mon[i]+1              
                             f_b39ye[i]<-tmp2$year[i]
                             f_b39de[i]<-tmp2$mday[i]} #97 values  
                             
for(i in which(f_b39mf==97)){f_b39mf[i]<-tmp2$mon[i]+1              
                             f_b39yf[i]<-tmp2$year[i]
                             f_b39df[i]<-tmp2$mday[i]} #97 values  

for(i in which(f_b39mg==97)){f_b39mg[i]<-tmp2$mon[i]+1              
                             f_b39yg[i]<-tmp2$year[i]
                             f_b39dg[i]<-tmp2$mday[i]} #97 values  
                             
                                                                 
                                                                                                  
                                                                                           
# Treatment of the "does not know" values

f_b38da[f_b38da>=98]<-NA
f_b38ma[f_b38ma>=98]<-NA
f_b38ya[f_b38ya>=98]<-NA   
f_b39da[f_b39da>=98]<-NA      
f_b39ma[f_b39ma>=98]<-NA  
f_b39ya[f_b39ya>=98]<-NA         

f_b38db[f_b38db>=98]<-NA
f_b38mb[f_b38mb>=98]<-NA
f_b38yb[f_b38yb>=98]<-NA         
f_b39db[f_b39db>=98]<-NA    
f_b39mb[f_b39mb>=98]<-NA  
f_b39yb[f_b39yb>=98]<-NA   

f_b38dc[f_b38dc>=98]<-NA
f_b38mc[f_b38mc>=98]<-NA
f_b38yc[f_b38yc>=98]<-NA   
f_b39dc[f_b39dc>=98]<-NA    
f_b39mc[f_b39mc>=98]<-NA  
f_b39yc[f_b39yc>=98]<-NA    

f_b38dd[f_b38dd>=98]<-NA
f_b38md[f_b38md>=98]<-NA
f_b38yd[f_b38yd>=98]<-NA                   
f_b39dd[f_b39dd>=98]<-NA    
f_b39md[f_b39md>=98]<-NA  
f_b39yd[f_b39yd>=98]<-NA  
       
f_b38de[f_b38de>=98]<-NA
f_b38me[f_b38me>=98]<-NA
f_b38ye[f_b38ye>=98]<-NA   
f_b39de[f_b39de>=98]<-NA   
f_b39me[f_b39me>=98]<-NA  
f_b39ye[f_b39ye>=98]<-NA   

f_b38df[f_b38df>=98]<-NA
f_b38mf[f_b38mf>=98]<-NA
f_b38yf[f_b38yf>=98]<-NA   
f_b39df[f_b39df>=98]<-NA   
f_b39mf[f_b39mf>=98]<-NA  
f_b39yf[f_b39yf>=98]<-NA      

f_b38dg[f_b38dg>=98]<-NA
f_b38mg[f_b38mg>=98]<-NA
f_b38yg[f_b38yg>=98]<-NA   
f_b39dg[f_b39dg>=98]<-NA   
f_b39mg[f_b39mg>=98]<-NA  
f_b39yg[f_b39yg>=98]<-NA   
                        
                  
#-------------------------------------------------------------------------------
# Computation of the number of days and weeks between the begin and the end of each attended vocational course 
#
# Remarks:
# f_b38ij=starting i(= day, month, year) of the respective vocational course (j=a,...,g), j=a corresponds to voc1, j=b to voc2,... j=g to voc7
# f_b39ij=stopping i(= day, month, year) of the respective vocational course (j=a,...,d)), j=a corresponds to voc1, j=b to voc2,... j=g to voc7
# Function period computes the number of days and week (see rows 91-101) from f_20ij and f_21ij
#-------------------------------------------------------------------------------

voc1_start<- paste(f_b38da,f_b38ma,f_b38ya, sep="/")
voc1_stop <- paste(f_b39da,f_b39ma,f_b39ya, sep="/")                                                                                                                                                               
voc1<-period(voc1_start,voc1_stop)
colnames(voc1)<- c("days_voc1", "weeks_voc1")

voc2_start<- paste(f_b38db,f_b38mb,f_b38yb, sep="/")
voc2_stop <- paste(f_b39db,f_b39mb,f_b39yb, sep="/")
voc2<-period(voc2_start,voc2_stop)  
colnames(voc2)<- c("days_voc2", "weeks_voc2")

voc3_start<- paste(f_b38dc,f_b38mc,f_b38yc, sep="/")
voc3_stop <- paste(f_b39dc,f_b39mc,f_b39yc, sep="/")
voc3<-period(voc3_start,voc3_stop)  
colnames(voc3)<- c("days_voc3", "weeks_voc3")

voc4_start<- paste(f_b38dd,f_b38md,f_b38yd, sep="/")
voc4_stop <- paste(f_b39dd,f_b39md,f_b39yd, sep="/")
voc4<-period(voc4_start,voc4_stop)  
colnames(voc4)<- c("days_voc4", "weeks_voc4")

voc5_start<- paste(f_b38de,f_b38me,f_b38ye, sep="/")
voc5_stop <- paste(f_b39de,f_b39me,f_b39ye, sep="/")
voc5<-period(voc5_start,voc5_stop)  
colnames(voc5)<- c("days_voc5", "weeks_voc5")

voc6_start<- paste(f_b38df,f_b38mf,f_b38yf, sep="/")
voc6_stop <- paste(f_b39df,f_b39mf,f_b39yf, sep="/")
voc6<-period(voc6_start,voc6_stop)  
colnames(voc6)<- c("days_voc6", "weeks_voc6")

voc7_start<- paste(f_b38dg,f_b38mg,f_b38yg, sep="/")
voc7_stop <- paste(f_b39dg,f_b39mg,f_b39yg, sep="/")
voc7<-period(voc7_start,voc7_stop)  
colnames(voc7)<- c("days_voc7", "weeks_voc7")

                              
                       
#Measurement errors suppression

#Negative values (due to programmes which began later than the 1st year after random assignement)
voc1[c(which(voc1$days_voc1<0)),]<-NA
voc2[c(which(voc2$days_voc2<0)),]<-NA
voc3[c(which(voc3$days_voc3<0)),]<-NA
voc4[c(which(voc4$days_voc4<0)),]<-NA
voc5[c(which(voc5$days_voc5<0)),]<-NA
voc6[c(which(voc6$days_voc6<0)),]<-NA
voc7[c(which(voc7$days_voc7<0)),]<-NA


#more than 365 days
voc1[c(which(voc1$days_voc1>365)),]<-NA
voc2[c(which(voc2$days_voc2>365)),]<-NA
voc3[c(which(voc3$days_voc3>365)),]<-NA
voc4[c(which(voc4$days_voc4>365)),]<-NA
voc5[c(which(voc5$days_voc5>365)),]<-NA
voc6[c(which(voc6$days_voc6>365)),]<-NA
voc7[c(which(voc7$days_voc7>365)),]<-NA

                            
#-------------------------------------------------------------------------------------------------------------

# Computation of the total hours spent in any vocational courses during the 1st year after random assignment

# Variables: 
# f_b43j = number of weeks where the individual did not attend vocational course 1,2,3,4 (1 corresponds to j=a, 2 to j=b, 3 to j=c, 4 to j=d)
# f_b44j = number of days per week in vocational courses 1,2,3,4 (j=a,b,c,d)
# f_b45j = number of hours per days in vocational courses 1,2,3,4 (j=a,b,c,d)
#-------------------------------------------------------------------------------------------------------------

# Voc 1
#---------

#Treatment of the 98 and 8 values
f_b43a[f_b43a>=98]<-NA
f_b43a[is.na(f_b43a)]<-0

f_b44a[f_b44a>=8]<-NA
f_b45a[f_b45a>=98]<-NA

a_hrs <- f_b44a*f_b45a  # hours per week in voc 1
hrsvoc1 <- a_hrs*(voc1$weeks_voc1-f_b43a) #total hours in voc 1 in the period where the individual took voc 1
        
     
# Voc 2
#---------

#Treatment of the 98 and 8 values
f_b43b[f_b43b>=98]<-NA
f_b43b[is.na(f_b43b)]<-0

f_b44b[f_b44b>=8]<-NA                
f_b45b[f_b45b>=98]<-NA

b_hrs <- f_b44b*f_b45b  # hours per week in voc 2 
hrsvoc2 <- b_hrs*(voc2$weeks_voc2-f_b43b) #total hours in voc 2 in the period where the individual took voc 2


# Voc 3
#---------

#Treatment of the 98 and 8 values
f_b43c[f_b43c>=98]<-NA
f_b43c[is.na(f_b43c)]<-0

f_b44c[f_b44c>=8]<-NA
f_b45c[f_b45c>=98]<-NA

c_hrs <- f_b44c*f_b45c   # hours per week in voc 3
hrsvoc3 <- c_hrs*(voc3$weeks_voc3-f_b43c) #total hours in voc 3 in the period where the individual took voc 3


# Voc 4
#---------

#Treatment of the 98 and 8 values
f_b43d[f_b43d>=98]<-NA
f_b43d[is.na(f_b43d)]<-0

f_b44d[f_b44d>=8]<-NA
f_b45d[f_b45d>=98]<-NA

d_hrs <- f_b44d*f_b45d   # hours per week in voc 4
hrsvoc4 <- d_hrs*(voc4$weeks_voc4-f_b43d)#total hours in voc 4 in the period where the individual took voc 4


# Voc 5
#---------

#Treatment of the 98 and 8 values
f_b43e[f_b43e>=98]<-NA
f_b43e[is.na(f_b43e)]<-0

f_b44e[f_b44e>=8]<-NA
f_b45e[f_b45e>=98]<-NA

e_hrs <- f_b44e*f_b45e   # hours per week in voc 5
hrsvoc5 <- e_hrs*(voc5$weeks_voc5-f_b43e)#total hours in voc 5 in the period where the individual took voc 5


# Voc 6
#---------

#Treatment of the 98 and 8 values
f_b43f[f_b43f>=98]<-NA
f_b43f[is.na(f_b43f)]<-0

f_b44f[f_b44f>=8]<-NA
f_b45f[f_b45f>=98]<-NA
                      
f_hrs <- f_b44f*f_b45f   # hours per week in voc 6
hrsvoc6 <- f_hrs*(voc6$weeks_voc6-f_b43f)#total hours in voc 6 in the period where the individual took voc 


# Voc 7
#---------

#Treatment of the 98 and 8 values
f_b43g[f_b43g>=98]<-NA
f_b43g[is.na(f_b43g)]<-0

f_b44g[f_b44g>=8]<-NA
f_b45g[f_b45g>=98]<-NA

g_hrs <- f_b44g*f_b45g   # hours per week in voc 7               
hrsvoc7 <- g_hrs*(voc7$weeks_voc7-f_b43g) #total hours in voc 7 in the period where the individual took voc 7



                   
# Measurement errors treated as NA
# Negative values (due to programmes which began later than the 1st year after random assignement)
hrsvoc1[hrsvoc1<0]<-NA
hrsvoc2[hrsvoc2<0]<-NA    
hrsvoc3[hrsvoc3<0]<-NA
hrsvoc4[hrsvoc4<0]<-NA
hrsvoc5[hrsvoc5<0]<-NA
hrsvoc6[hrsvoc6<0]<-NA
hrsvoc7[hrsvoc7<0]<-NA
             
                                                      
# Dataframe with of all hours spent in vocational courses (periods 1-4)
#----------------------------------------------------------------------
hrs_voc <- data.frame(id,
                      hrsvoc1, 
                      hrsvoc2,
                      hrsvoc3,
                      hrsvoc4, 
                      hrsvoc5,
                      hrsvoc6,
                      hrsvoc7)
                                                                 
# Total hours spent in any vocational courses during the 1st year    
#-----------------------------------------------------------------  
S <- function(x) sum(x, na.rm=TRUE)                        
hrs_voc$total_hrs_vocy1  <- apply(hrs_voc[,-1], 1, S)
                                            
#--------------------------------------------------------------------------------------------------------------
# Total hours spent either in academics or vocational classes in the 12 months following the random assignment
#--------------------------------------------------------------------------------------------------------------
tot_vocacad_y1<- data.frame("total_hrs_acady1"=hrs_acad$total_hrs_acady1,"total_hrs_vocy1"=hrs_voc$total_hrs_vocy1)
tot_vocacad_y1$totalhrs_acadvocy1<- apply(tot_vocacad_y1, 1, S)


# Treatment variables dataframe
T <- data.frame( id     ,
                 "total_hrs_acady1"=hrs_acad$total_hrs_acady1,
                 "total_hrs_vocy1"=hrs_voc$total_hrs_vocy1,
                 "totalhrs_acadvocy1"=tot_vocacad_y1$totalhrs_acadvocy1
                 )
                        

#------------------------------------------------------------------------------
# Mediator     
#------------------------------------------------------------------------------

M2 <- pworky2                          # Prop of wks employed in year 2
M3 <- pworky3                          # Prop of wks employed in year 3
M4 <- pworky4


# Mediator dataframe 
Med <- data.frame(id, M2, M3) # add the identifier


# Merging the data

wdf<-merge(T,Med,by="id")

#------------------------------------------------------------------------------
# Outcome 
#------------------------------------------------------------------------------

# (a) Number of arrests in year 4
Y_arr <- narry4       

# (b) used any drugs in the month before 48 months interview
Y_dr <- anydr48

# Outcome dataframe
Y <- data.frame(id, Y_arr, Y_dr)

# Merging the data
wdf<-merge(wdf,Y,by="id")

#------------------------------------------------------------------------------
# Pre-treatment Covariates 
#------------------------------------------------------------------------------

# a) Demographics
#-----------------

# Gender
X <- data.frame(id,female)  # 0 if male, 1 if female

# Age, age squared, age cubed
X$age  <- age_cat
X$age2 <- age_cat^2
X$age3 <- age_cat^3

# Race
X$race <- race_eth   # 1=white, 2=black, 3=hispanic, 4=Indian 

X$race_white     <- indicator(race_eth==1) # dummy for category 1
X$race_black     <- indicator(race_eth==2) # dummy for category 2 
X$race_hispanic  <- indicator(race_eth==3) # dummy for category 3
X$race_indian    <- indicator(race_eth==4) # dummy for category 4 

                    
# Nonresidential slot
X$nonres <- nonres      # 1 if designated for a nonresidential slot

# b) Education
#---------------
                                          
# Diploma
X$educ <- educ_gr   # 1 if no hs diploma or ged (general education diploma, equivalent to the hs diploma), 2 if ged, 3 if hs diploma
X$educ_nodiploma        <- indicator(educ_gr==1) # dummy for category 1
X$educ_geddiploma       <- indicator(educ_gr==2) # dummy for category 2 
X$educ_hsdiploma        <- indicator(educ_gr==3) # dummy for category 3 
                                     
# Highest grade completed and its squared : [0,20], 98 or 99 if not answered 
X$hgrd  <- b1          # What is the highest grade you have already completed?
X$hgrd[X$hgrd>=98]<-NA # Replace the 98-99 answers by NA 

# Highest grade completed squared
X$hgrd2<-X$hgrd^2 


# Native language
X$ntv_lang <- j10             # 1=English, 2=Spanish, 3=Asian language, 4=French
X$ntv_lang[X$ntv_lang>=8]<-NA # Replace the 8-9 answers by NA 


X$ntv_engl  <- indicator(X$ntv_lang==1) # dummy for category 1
X$ntv_span  <- indicator(X$ntv_lang==2) # dummy for category 2
X$ntv_asian <- indicator(X$ntv_lang==3) # dummy for category 3   
X$ntv_fr    <- indicator(X$ntv_lang==4) # dummy for category 4


# In a training program since 1 year before randomisation date 

X$trainingbefore <- d2  
X$trainingbefore[X$trainingbefore>=8]<-NA # Replace the 8 answers by NA 



# c) Earnings
#---------------
 
# Average Weekly Earnings over all the jobs the individual had
weekly_earn <- 
         data.frame(c141,     #weekly earnings on job 1             
                    c142,     #weekly earnings on job 2
                    c143,     #weekly earnings on job 3                  
                    c144,    #weekly earnings on job 4
                    c145,     #weekly earnings on job 5
                    c146,     #weekly earnings on job 6
                    c147,     #weekly earnings on job 7
                    c148,     #weekly earnings on job 8   (only NAs)
                    c149,     #weekly earnings on job 9   (only NAs)
                    c140)     #weekly earnings on job 10  (only NAs)
                    
                    
sum_earnings<-apply(weekly_earn,1,sum, na.rm=TRUE)

A<-function(x) sum(!is.na(x))   #function used to count number of jobs                             
divisor<- apply(weekly_earn, 1, A)  

mean_weekearn<- numeric()

for(i in 1: nrow(data)){
if(divisor[i] != 0){mean_weekearn[i]<- sum_earnings[i]/divisor[i]} else{mean_weekearn[i] <-0}}

X$mwearn<- mean_weekearn


X$mwearn[X$mwearn>5999]<-NA # Delete the 4 observations which are probably measurement errors.

# Ever worked
X$everwkd <- c2     # 1 if ever worked, 0 otherwise
X$everwkd[X$everwkd>=8] <- NA # Replace the 8-9 answers by NA 

# Type of area (1=superdns,2=dns,3=nondns area at randdt)
X$typearea <- typearea

X$typearea_superdns    <- indicator(typearea==1) # dummy for category 1
X$typearea_dns         <- indicator(typearea==2) # dummy for category 2 
X$typearea_nondns      <- indicator(typearea==3) # dummy for category 3 

                                                       
# Non-salarial income                                 
X$g1<-g1  # recvd afdc (Aid to Families with Dependent Children) since 1 yr prior to randdt 
X$g1[X$g1>=8]<- NA # Replace the 8-9 answers by NA

X$g2<-g2  #recvd afdc every month since ref date
X$g2[X$g2>=8]<- NA # Replace the 8-9 answers by NA 

X$g4<-g4  #got other public assist.: ga, ssi, ssa
X$g4[X$g4>=8]<- NA # Replace the 8-9 answers by NA 

X$g5<-g5  #got public assist. every month
X$g5[X$g5>=8]<- NA # Replace the 8-9 answers by NA 

X$g7<-g7  #recvd food stamps since ref date
X$g7[X$g7>=8]<- NA # Replace the 8-9 answers by NA 

X$g8<-g8  #got food stamps every month
X$g8[X$g8>=8]<- NA # Replace the 8-9 answers by NA 

# Last year income
X$g10<-g10  #tot income from all household members last yr
X$g10[X$g10>=8]<- NA # Replace the 8-9 answers by NA 

X$g12<-g12  #total personal income last year (ordinal)
X$g12[X$g12>=8]<- NA # Replace the 8-9 answers by NA 


# d) Household characteristics 
#------------------------------

# Children at random assignment
X$haschldY0 <- haschld  # 0 if no child, 1 if one child or more

# Marital status on randomization date
X$marstat0 <- f20   #1=married, 2=separated, 3= divorced, 4=widowed, 5=living together unmarried, 6=never married, and not living together unmarried
X$marstat_married        <- indicator(f20==1) # dummy for category 1
X$marstat_separated      <- indicator(f20==2) # dummy for category 2  
X$marstat_divorced       <- indicator(f20==3) # dummy for category 3
X$marstat_widowed        <- indicator(f20==4) # dummy for category 4
X$marstat_livetogunm     <- indicator(f20==5) # dummy for category 5
X$marstat_notlivetogunm  <- indicator(f20==6) # dummy for category 6

#Head of household on randomization date (are you Head of household ?)
X$hohhd0 <- f21 # 1 if yes, 0 otherwise.   
X$hohhd0[X$hohhd0>=8]<- NA # Replace the 8-9 answers by NA 
                    
# Number of people you lived with on rand date
X$peopleathome <- f1    
X$peopleathome[X$peopleathome>=98]<- NA # Replace the 98-99 answers by NA 
            
# Live with how many (biological) parents on randomization date

f3var<-paste("f3",letters[1:15],sep="")   # relationship to person j (j=a,...,o)
f4var<-paste("f4",letters[1:15],sep="")   # sex of person j (j=a,...,o) (1=male, 0=female)

parents <- matrix(,nrow=nrow(data),ncol=(length(f3var)))

for(i in 1:nrow(data)){
  for(j in 1:length(f3var)){
    if(  data[i,match(f3var[j],colnames(data))]==1
             & !is.na(data[i,match(f3var[j],colnames(data))])
             & !is.na(data[i,match(f4var[j],colnames(data))])){
               parents[i,j] <- data[i,match(f4var[j],colnames(data))]

                }else{parents[i,j]<-NA}
  }
 
}

parents<-data.frame(parents)
colnames(parents)<-f3var

# Recode 0 for male into 4
parents_reval<-parents
parents_reval[parents_reval==0]<-4

#living with 2 parents (livewpar=1)
livewpar<-numeric()

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
       if(sum(!is.na(parents_reval[i,])==2
     &    sum(parents_reval[i,], na.rm=TRUE)==5)){
                                                                                                     
             livewpar[i]<-1
             } else{livewpar[i]<-NA}}}
                                     
            
# living with only 1 male parent (livewpar=2)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
       if(sum(!is.na(parents_reval[i,])==1
     &    sum(parents_reval[i,], na.rm=TRUE)==4)){
                                                                                                     
             livewpar[i]<-2
             }}}
             
            
# living with 1 female parent (livewpar=3)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
       if(sum(!is.na(parents_reval[i,])==1
     &    sum(parents_reval[i,], na.rm=TRUE)==1)){
                                                                                                     
             livewpar[i]<-3
             }}}

# living without parent (livewpar=0)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
       if(sum(!is.na(parents_reval[i,]))==0){
                                                                                                     
             livewpar[i]<-0
             }}}
             
# living with 2 male parents (livewpar=4)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
      if(sum(!is.na(parents_reval[i,])==2
     &    sum(parents_reval[i,], na.rm=TRUE)==8)){                                                                                                     
             livewpar[i]<-4
             }}}
             
# living with 2 female parents (livewpar=5)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
      if(sum(!is.na(parents_reval[i,])==2
     &    sum(parents_reval[i,], na.rm=TRUE)==2)){                                                                                                     
             livewpar[i]<-5
             }}}
             
# living with 2 female parents and 1 male parent (livewpar=6)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
      if(sum(!is.na(parents_reval[i,])==3
     &    sum(parents_reval[i,], na.rm=TRUE)==6)){                                                                                                     
             livewpar[i]<-6
             }}}
             
            
# living with 2 male parents and 1 female parent (livewpar=7)     

for(i in 1:nrow(data)){
   for(j in 1:length(f3var)){
      if(sum(!is.na(parents_reval[i,])==3
     &    sum(parents_reval[i,], na.rm=TRUE)==9)){                                                                                                     
             livewpar[i]<-7
             }}}
             
            
                  
X$livewpar<-livewpar                                            
           
X$par0           <- indicator(X$livewpar==0) # dummy for category 0    
X$par2mf         <- indicator(X$livewpar==1) # dummy for category 1    
X$par1m          <- indicator(X$livewpar==2) # dummy for category 2  
X$par1f          <- indicator(X$livewpar==3) # dummy for category 3
X$par2mm         <- indicator(X$livewpar==4) # dummy for category 4
X$par2ff         <- indicator(X$livewpar==5) # dummy for category 5
X$par3ffm        <- indicator(X$livewpar==6) # dummy for category 6
X$par3mmf        <- indicator(X$livewpar==7) # dummy for category 7
           
# e) Family background
#--------------------------
                          
# Who was the household head when you were 14 (Cf. DataDoc_JC_Volume_I.pdf p. 115 ss.)
X$hohhd14 <- f25  
X$hohhd14[X$hohhd14>=98]<- NA # Replace the 98-99 answers by NA 

X$hohhd14_other          <-   indicator(f25==0) # dummy for category 0
X$hohhd14_father         <-   indicator(f25==1) # dummy for category 1   
X$hohhd14_mother         <-   indicator(f25==2) # dummy for category 2  
X$hohhd14_bothpar        <-   indicator(f25==3) # dummy for category 3
X$hohhd14_stepmoth       <-   indicator(f25==4) # dummy for category 4
X$hohhd14_stepfath       <-   indicator(f25==5) # dummy for category 5
X$hohhd14_fosterpar      <-   indicator(f25==6) # dummy for category 6
X$hohhd14_auntuncle      <-   indicator(f25==7) # dummy for category 7
X$hohhd14_grandpar       <-   indicator(f25==8) # dummy for category 8  
X$hohhd14_sibling        <-   indicator(f25==9) # dummy for category 9
X$hohhd14_nephew         <-   indicator(f25==10)# dummy for category 10
X$hohhd14_cousin         <-   indicator(f25==11)# dummy for category 11
X$hohhd14_husbwife       <-   indicator(f25==12)# dummy for category 12 
X$hohhd14_boygirlfriend  <-   indicator(f25==13)# dummy for category 13  
X$hohhd14_otherrelative  <-   indicator(f25==14)# dummy for category 14
X$hohhd14_nonrelative    <-   indicator(f25==15)# dummy for category 15 
                              

                                                                                 
# Highest grade completed of mother (cf. DataDoc_JC_Volume_I.pdf p. 115 ss.)
X$hgrd_mum <- f26
X$hgrd_mum[X$hgrd_mum>=97]<- NA # Replace the 97-98-99 answers by NA 

               
# When 14 type of work mother did cf. DataDoc_JC_Volume_I.pdf p. 117 ss.)
X$work_mum <- f27
X$work_mum[X$work_mum>=98]<- NA # Replace the 98-99 answers by NA 

X$work_mum_other           <-   indicator(f27==0) # dummy for category 0                 
X$work_mum_laborer         <-   indicator(f27==1) # dummy for category 1   
X$work_mum_manager         <-   indicator(f27==2) # dummy for category 2  
X$work_mum_military        <-   indicator(f27==3) # dummy for category 3
X$work_mum_officeworker    <-   indicator(f27==4) # dummy for category 4
X$work_mum_operator        <-   indicator(f27==5) # dummy for category 5
X$work_mum_owner           <-   indicator(f27==6) # dummy for category 6
X$work_mum_professionalnoT <-   indicator(f27==7) # dummy for category 7
X$work_mum_professional    <-   indicator(f27==8) # dummy for category 8
X$work_mum_protectiveserv  <-   indicator(f27==9) # dummy for category 9  
X$work_mum_sales           <-   indicator(f27==10)# dummy for category 10
X$work_mum_serviceworker   <-   indicator(f27==11)# dummy for category 11
X$work_mum_teachingprof    <-   indicator(f27==12)# dummy for category 12
X$work_mum_technical       <-   indicator(f27==13)# dummy for category 13  
X$work_mum_tradesperson    <-   indicator(f27==14)# dummy for category 14
X$work_mum_farmer          <-   indicator(f27==15)# dummy for category 15 
X$work_mum_homemaker       <-   indicator(f27==16)# dummy for category 16 
X$work_mum_didnotwork      <-   indicator(f27==96)# dummy for category 96 
X$work_mum_deceased        <-   indicator(f27==97)# dummy for category 97 
                   
                                                                        

# Highest grade completed of father (cf. DataDoc_JC_Volume_I.pdf p. 119 ss.)
X$hgrd_dad <- f28
X$hgrd_dad[X$hgrd_dad>=97]<- NA # Replace the 97-98-99 answers by NA 

# When 14 type of work father did cf. DataDoc_JC_Volume_I.pdf p. 121 ss.)
X$work_dad <- f29    
X$work_dad[X$work_dad>=98]<- NA # Replace the 98-99 answers by NA 

X$work_dad_other           <-   indicator(f29==0) # dummy for category 0                 
X$work_dad_laborer         <-   indicator(f29==1) # dummy for category 1   
X$work_dad_manager         <-   indicator(f29==2) # dummy for category 2  
X$work_dad_military        <-   indicator(f29==3) # dummy for category 3
X$work_dad_officeworker    <-   indicator(f29==4) # dummy for category 4
X$work_dad_operator        <-   indicator(f29==5) # dummy for category 5
X$work_dad_owner           <-   indicator(f29==6) # dummy for category 6
X$work_dad_professionalnoT <-   indicator(f29==7) # dummy for category 7
X$work_dad_professional    <-   indicator(f29==8) # dummy for category 8
X$work_dad_protectiveserv  <-   indicator(f29==9) # dummy for category 9  
X$work_dad_sales           <-   indicator(f29==10)# dummy for category 10
X$work_dad_serviceworker   <-   indicator(f29==11)# dummy for category 11
X$work_dad_teachingprof    <-   indicator(f29==12)# dummy for category 12
X$work_dad_technical       <-   indicator(f29==13)# dummy for category 13  
X$work_dad_tradesperson    <-   indicator(f29==14)# dummy for category 14
X$work_dad_farmer          <-   indicator(f29==15)# dummy for category 15 
X$work_dad_homemaker       <-   indicator(f29==16)# dummy for category 16 
X$work_dad_didnotwork      <-   indicator(f29==96)# dummy for category 96 
X$work_dad_deceased        <-   indicator(f29==97)# dummy for category 97 
                   


# How often got welfare while growing up
X$welfare_child <- f30 
X$welfare_child[X$welfare_child>=8]<- NA # Replace the 8-9 answers by NA 


# Birthorder of the observation among the siblings at home (living with at the interview)
f3var<-paste("f3",letters[1:15],sep="")   # relationship to person j (j=a,...,o)
f6var<-paste("f6",letters[1:15],sep="")   # age of person j (j=a,...,o)
ages <- matrix(,nrow=nrow(data),ncol=(length(f3var)))

for(i in 1:nrow(data)){
  for(j in 1:length(f3var)){
    if(  data[i,match(f3var[j],colnames(data))]==6
             & !is.na(data[i,match(f3var[j],colnames(data))])
             & data[i,match(f6var[j],colnames(data))]!=998
             & data[i,match(f6var[j],colnames(data))]!=999
             & !is.na(data[i,match(f6var[j],colnames(data))])){
               ages[i,j] <- data[i,match(f6var[j],colnames(data))]

                }else{ages[i,j]<-NA}
  }
  
}

colnames(ages)<-f6var
ages <- data.frame(cbind(ages,age_cat))

prank<-function(x){                #Function to rank the ages (1= smallest,...)
  r<-rank(-x)
  r[is.na(x)]<-NA
  r
}

ages_ord<- data.frame(t(apply(ages,1, prank)))

colnames(ages_ord)[16]<-"birthorder" #if x.5 => twin with the xth birthorder.

ages<-cbind(ages,"birthorder"=ages_ord$birthorder)

X$birthorder<- ages_ord$birthorder


# f) Health status  
#------------------
X$h1<-h1    # general health status
X$h1[X$h1>=8]<- NA # Replace the 8-9 answers by NA 

X$h1_excel    <- indicator(h1==1) # dummy for category 1
X$h1_good     <- indicator(h1==2) # dummy for category 2
X$h1_fair     <- indicator(h1==3) # dummy for category 3
X$h1_poor     <- indicator(h1==4) # dummy for category 4

# Merge h1_fair with h1_poor to avoid categories which are too small
X$h1_fair_poor<- X$h1_fair+ X$h1_poor #dummy for either poor or fair health


X$h2<-h2    # had physical/emotional problems
X$h2[X$h2>=8]<- NA # Replace the 8-9 answers by NA 

X$h4<-h4    # ever smoked
X$h4[X$h4>=8]<- NA # Replace the 8-9 answers by NA 

X$h5<-h5    # how often smoked last year
X$h5[X$h5>=8]<- NA # Replace the 8-9 answers by NA 

X$h6<-h6    # ever used alcohol
X$h6[X$h6>=8]<- NA # Replace the 8-9 answers by NA 

X$h7<-h7    # how often used alcohol in last year
X$h7[X$h7>=8]<- NA # Replace the 8-9 answers by NA 

X$h9<-h9    # ever used marijuana or hashish
X$h9[X$h9>=8]<- NA # Replace the 8-9 answers by NA 

X$h10<-h10  # how often used marijuana in last year 
X$h10[X$h10>=8]<- NA # Replace the 8-9 answers by NA 

X$h12<-h12  # ever snorted cocaine powder 
X$h12[X$h12>=8]<- NA # Replace the 8-9 answers by NA 

X$h13<-h13  # how often snorted cocaine in last year
X$h13[X$h13>=8]<- NA # Replace the 8-9 answers by NA 

X$h15<-h15  # ever smoked crack cocaine or freebased
X$h15[X$h15>=8]<- NA # Replace the 8-9 answers by NA 

X$h16<-h16  # how often smoked crack or freebased in last year
X$h16[X$h16>=8]<- NA # Replace the 8-9 answers by NA 

X$h18<-h18  # ever used heroin/opium
X$h18[X$h18>=8]<- NA # Replace the 8-9 answers by NA 

X$h19<-h19  # how often used heroin in last year

X$h21<-h21  # ever used speed
X$h21[X$h21>=8]<- NA # Replace the 8-9 answers by NA 

X$h22<-h22  # how often used methamphetamines/speed/uppers in last year
X$h22[X$h22>=8]<- NA # Replace the 8-9 answers by NA 

X$h24<-h24  # ever used lsd/peyote/psilocybin/hallucinogenic drugs to get high
X$h24[X$h24>=8]<- NA # Replace the 8-9 answers by NA 

X$h25<-h25  # how often used lsd/peyote/psilocybin/hallucinogenic drugs in last year
X$h25[X$h25>=8]<- NA # Replace the 8-9 answers by NA 

X$h27<-h27  # ever shot or injected drugs
X$h27[X$h27>=8]<- NA # Replace the 8-9 answers by NA 

X$h28<-h28  # how often injected drugs in last year
X$h28[X$h28>=8]<- NA # Replace the 8-9 answers by NA 

X$h29<-h29  # ever used other illegal drugs
X$h29[X$h29>=6]<- NA # Replace the 7-8-9 answers by NA #CORRECTED

X$h30a<-h30a# how often used other drugs in last year
X$h30a[X$h30a>=8]<- NA # Replace the 8-9 answers by NA 
                                                    
X$h31<-h31  # ever in a drug/alcohol treatment program
X$h31[X$h31>=8]<- NA # Replace the 8-9 answers by NA 

X$h33b<-h33b# how long in drug treatment program-unit


# g) Arrest behaviour
#----------------------
X$i1<-i1     # ever arrested                               
X$i1[X$i1>=8]<- NA # Replace the 8-9 answers by NA 

X$i2<-i2     # number of times arrested
X$i2[X$i2>=98]<- NA # Replace the 98-99 answers by NA 


# number of times served time in prison
dfi10<-data.frame(i10a,i10b,i10c,i10d,i10e)
dfi10[dfi10 == 8] <- NA
dfi10[dfi10 == 9] <- NA
S<- function(x) sum(x, na.rm=TRUE)
X$i10<-apply(dfi10,1,S)

# h) Knowledge of JC and Recruiting Experience:
#-----------------------------------------------
X$e1<-e1   # how did you 1st hear about job corps 
X$e1[X$e1>=98]<- NA # Replace the 98-99 answers by NA 


X$e1_other           <-   indicator(e1==0) # dummy for category 0                 
X$e1_parents         <-   indicator(e1==1) # dummy for category 1   
X$e1_otherrel        <-   indicator(e1==2) # dummy for category 2  
X$e1_friends         <-   indicator(e1==3) # dummy for category 3
X$e1_media           <-   indicator(e1==4) # dummy for category 4
X$e1_school          <-   indicator(e1==5) # dummy for category 5
X$e1_WICS            <-   indicator(e1==6) # dummy for category 6
X$e1_employserv      <-   indicator(e1==7) # dummy for category 7
X$e1_union           <-   indicator(e1==8) # dummy for category 8
X$e1_welfareoffic    <-   indicator(e1==9) # dummy for category 9  
X$e1_paroleoffice    <-   indicator(e1==10)# dummy for category 10
X$e1_directcontJC    <-   indicator(e1==11)# dummy for category 11
X$e1_livepresentrecr <-   indicator(e1==12)# dummy for category 12
X$e1_jtpa            <-   indicator(e1==13)# dummy for category 13  
X$e1_church          <-   indicator(e1==14)# dummy for category 14
                                    

X$e3<-e3   # did parent/friend attend job corps  
X$e3[X$e3>=8]<- NA # Replace the 8-9 answers by NA 

X$e4<-e4   # knew anyone who attended job corps  
X$e4[X$e4>=8]<- NA # Replace the 8-9 answers by NA 

X$e5<-e5   # where got most info about job corps   
X$e5[X$e5>=98]<- NA # Replace the 98-99 answers by NA  

                   

X$e5_other           <-   indicator(e5==0) # dummy for category 0                 
X$e5_parents         <-   indicator(e5==1) # dummy for category 1   
X$e5_otherrel        <-   indicator(e5==2) # dummy for category 2  
X$e5_friends         <-   indicator(e5==3) # dummy for category 3
X$e5_media           <-   indicator(e5==4) # dummy for category 4
X$e5_school          <-   indicator(e5==5) # dummy for category 5
X$e5_WICS            <-   indicator(e5==6) # dummy for category 6
X$e5_employserv      <-   indicator(e5==7) # dummy for category 7
X$e5_union           <-   indicator(e5==8) # dummy for category 8
X$e5_welfareoffic    <-   indicator(e5==9) # dummy for category 9  
X$e5_paroleoffice    <-   indicator(e5==10)# dummy for category 10
X$e5_directcontJC    <-   indicator(e5==11)# dummy for category 11
X$e5_livepresentrecr <-   indicator(e5==12)# dummy for category 12
X$e5_jtpa            <-   indicator(e5==13)# dummy for category 13  
X$e5_church          <-   indicator(e5==14)# dummy for category 14
     



X$e6<-e6   # 1st spoke by tel/in-person to recruiter  
X$e6[X$e6>=8]<- NA # Replace the 8-9 answers by NA 

X$e6_byphone         <-   indicator(e6==1) # dummy for category 1   
X$e6_inperson        <-   indicator(e6==2) # dummy for category 2  


X$e7<-e7   # did you call or did recruiter call 
X$e7[X$e7>=8]<- NA # Replace the 8-9 answers by NA 

X$e7_recrcalled           <-   indicator(e7==1) # dummy for category 1   
X$e7_samplemembcalled     <-   indicator(e7==2) # dummy for category 2     
X$e7_samplemembcalled800  <-   indicator(e7==3) # dummy for category 3     
                                
X$e8<-e8   # where 1st spoke to recruiter  
X$e8[X$e8>=8]<- NA # Replace the 8-9 answers by NA   
         
X$e8_other              <-   indicator(e8==0) # dummy for category 7
X$e8_JCcenter           <-   indicator(e8==1) # dummy for category 1   
X$e8_recruitersoffice   <-   indicator(e8==2) # dummy for category 2     
X$e8_employservoffice   <-   indicator(e8==3) # dummy for category 3       
X$e8_WICSoffice         <-   indicator(e8==4) # dummy for category 4     
X$e8_athome             <-   indicator(e8==5) # dummy for category 5
X$e8_atschool           <-   indicator(e8==6) # dummy for category 6
X$e8_church             <-   indicator(e8==7) # dummy for category 7
                                                                                                                         

#About how long did the JC recruiter say you were expected to stay in JC ?
e9e[e9e==98]<-NA                                                                                              
e9f[e9f==8]<-NA                    
df_e9ef<- data.frame(e9e, "e9f_y"=e9f*12 )
df_e9ef$e9ef<-apply(df_e9ef,1, S)
X$e9ef<-df_e9ef$e9ef  # Total in months (the 0 are either NA or 98 or 8 values of e9e and e9f)            


X$e10<-e10 # did recruiter say when you could leave?
X$e10[X$e10>=6]<- NA # Replace the 6-7-8-9 answers by NA   


X$e12<-e12 #how much time recruiter spent with you
X$e12[X$e12>=8]<- NA # Replace the 8-9 answers by NA   

                    

# Did you talk about going to JC with your...

X$e13a<-e13a     # parents or guardians
X$e13a[X$e13a>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13b<-e13b     # other relatives 
X$e13b[X$e13>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13c<-e13c     # friends
X$e13c[X$e13c>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13d<-e13d     # school teacher or counsellor
X$e13d[X$e13d>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13e<-e13e     # caseworker
X$e13e[X$e13e>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13f<-e13f     # probation officer
X$e13f[X$e13f>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13g<-e13g     # church leader
X$e13g[X$e13g>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e13h<-e13h     # another adult
X$e13h[X$e13h>=7]<- NA # Replace the 7-8-9 answers by NA   



#Was PERSON'S advice important to your decisions to go to JC ?

X$e14a<-e14a    # parents or guardians
X$e14a[X$e14a>=8]<- NA # Replace the 8-9 answers by NA   

X$e14b<-e14b    # other relatives
X$e14b[X$e14b>=8]<- NA # Replace the 8-9 answers by NA   

X$e14c<-e14c    # friends
X$e14c[X$e14c>=8]<- NA # Replace the 8-9 answers by NA   

X$e14d<-e14d    # school teacher or counsellor
X$e14d[X$e14d>=8]<- NA # Replace the 8-9 answers by NA   

X$e14e<-e14e    # caseworker
X$e14e[X$e14e>=8]<- NA # Replace the 8-9 answers by NA   

X$e14f<-e14f    # probation officer
X$e14f[X$e14f>=8]<- NA # Replace the 8-9 answers by NA   

X$e14g<-e14g    # church leader
X$e14g[X$e14g>=8]<- NA # Replace the 8-9 answers by NA   

X$e14h<-e14h    # another adult
X$e14h[X$e14h>=8]<- NA # Replace the 8-9 answers by NA   

                  
  

#Did (PERSON of e14) encourage or discourage you to go to JC ? 

X$e15a<-e15a    # parents or guardians
X$e15a[X$e15a>=8]<- NA # Replace the 8-9 answers by NA   

X$e15b<-e15b    # other relatives
X$e15b[X$e15b>=8]<- NA # Replace the 8-9 answers by NA   

X$e15c<-e15c    # friends
X$e15c[X$e15c>=8]<- NA # Replace the 8-9 answers by NA   

X$e15d<-e15d    # school teacher or counsellor
X$e15d[X$e15d>=8]<- NA # Replace the 8-9 answers by NA   

X$e15e<-e15e    # caseworker
X$e15e[X$e15e>=8]<- NA # Replace the 8-9 answers by NA   

X$e15f<-e15f    # probation officer
X$e15f[X$e15f>=8]<- NA # Replace the 8-9 answers by NA   

X$e15g<-e15g    # church leader
X$e15g[X$e15g>=8]<- NA # Replace the 8-9 answers by NA   

X$e15h<-e15h    # another adult
X$e15h[X$e15h>=8]<- NA # Replace the 8-9 answers by NA   

                                                      
X$e16   <- e16  # How much did recruiter encourage you
X$e16[X$e16>=8]<- NA # Replace the 8-9 answers by NA   

X$e16a  <- e16a # did you persuade jc recruiter to go  
X$e16a[X$e16a>=8]<- NA # Replace the 8-9 answers by NA   

                  
# Importance of some reasons to go to JC
X$e17a<-e17a    # getting away from home
X$e17a[X$e17a>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17b<-e17b    # getting away from problems in your community, such as crime or violence
X$e17b[X$e17b>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17c<-e17c    # getting job training
X$e17c[X$e17c>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17d<-e17d    # desire to achieve a career goal
X$e17d[X$e17d>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17e<-e17e    # getting a GED
X$e17e[X$e17e>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17f<-e17f    # not being able to find work
X$e17f[X$e17f>=7]<- NA # Replace the 7-8-9 answers by NA   

X$e17g<-e17g    # other personal reasons   
X$e17g[X$e17g>=7]<- NA # Replace the 7-8-9 answers by NA   


X$e20 <- e20  # Most important reason to join JC
X$e20[X$e20>=8]<- NA # Replace the 8-9 answers by NA              
         
X$e20_getawayfromhome         <-   indicator(e20==1) # dummy for category 1   
X$e20_getawayfromcrime        <-   indicator(e20==2) # dummy for category 2  
X$e20_getjobtraining          <-   indicator(e20==3) # dummy for category 3
X$e20_getGED                  <-   indicator(e20==4) # dummy for category 4
X$e20_unablefindwork          <-   indicator(e20==5) # dummy for category 5
X$e20_otherpersreason         <-   indicator(e20==6) # dummy for category 6
X$e20_achievingcareergoal     <-   indicator(e20==7) # dummy for category 7
     

X$e21 <- e21  # Knew what type of training wanted
X$e21[X$e21>=8]<- NA # Replace the 8-9 answers by NA   

X$e23 <- e23  # Chances of getting JC training in the trade you wanted (excellent, good,fair, poor)     
X$e23[X$e23>=7]<- NA # Replace the 8-9 answers by NA   

                                        
#expected hourly earnings after jc (USD)                 
e24cents<-e24c/100
X$e24usd<-e24+e24cents

X$e24usd[X$e24usd>=97]<- NA # Replace the 97-98-99 answers by NA  

X$e25<-e25 #knew which center wanted to go to before the application to JC
X$e25[X$e25>=8]<- NA # Replace the 8-9 answers by NA   
  
X$e28 <-  e28 # Recruiter tells about the wait time until enrollment
X$e28[X$e28>=8]<- NA # Replace the 8-9 answers by NA   


#Total waiting time in weeks
e29f[e29f==98]<-NA
df_wait<-data.frame(e29e, e29f*4)
df_wait$e29<- apply(df_wait,1, S)
X$e29<-df_wait$e29


X$e30 <-  e30 # extent to which jc will help math skills
X$e30[X$e30>=8]<- NA # Replace the 8-9 answers by NA   

X$e31 <-  e31 # extent to which jc will help reading skills
X$e31[X$e31>=8]<- NA # Replace the 8-9 answers by NA   

X$e32 <-  e32 # extent to which jc will help social skills
X$e32[X$e32>=8]<- NA # Replace the 8-9 answers by NA   

X$e33 <-  e33 # extent to which jc will help self-control
X$e33[X$e33>=8]<- NA # Replace the 8-9 answers by NA   

X$e34 <-  e34 # extent to which jc will help self-esteem
X$e34[X$e34>=8]<- NA # Replace the 8-9 answers by NA   

X$e35 <-  e35 # extent to which jc will help provide training for a specific job
X$e35[X$e35>=8]<- NA # Replace the 8-9 answers by NA   

X$e36 <-  e36 # extent to which jc will help lead to new friendships
X$e36[X$e36>=8]<- NA # Replace the 8-9 answers by NA   

X$e37 <-  e37 # worried about JC
X$e37[X$e37>=8]<- NA # Replace the 8-9 answers by NA   

  

# Missing dummies construction

Xmissdummies<-missdum(X) 
colnames(Xmissdummies)<-paste(colnames(X),"missdum", sep="")
colnames(Xmissdummies)[which(names(Xmissdummies) == "idmissdum")] <- "id"
Xmissdummies$id<-id


# Set the X missing values to 0
for(j in 1:ncol(X)){
  for (i in 1: nrow(X)){
      if(is.na(X[i,j])){
        X[i,j]<-0}
}}

# Merge both X dataframes
X<-merge(X,Xmissdummies,by="id")

# Merging the data
jcUNIFR<-merge(wdf,X,by="id")           

# Save the final dataset
save(jcUNIFR, file = paste (ddpath ,  "my_jcUNIFR.RData",  sep=""))

#Detach the data                          
detach(data)



