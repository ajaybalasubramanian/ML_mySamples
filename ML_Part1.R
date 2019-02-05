library(klaR)
library(caret)


wdat <- read.table("C:/Users/mythi/ML\ Ajay/CS498MachineLearning-master/CS498MachineLearning-master/HW1/pima-indians-diabetes.csv", sep = ",")
valueX<-wdat[,-c(9)]
valueY<-wdat[,9]
trscore <- array(dim = 10)
tescore <- array(dim = 10)


#####################
#Part A
#############
for (wi in 1:10) {
  wtd <- createDataPartition(y = valueY, p = 0.8, list = FALSE) # 80% of the data into training
  nbx <- valueX                                 # matrix of features
  ntrbx <- nbx[wtd, ]                         # training features
  ntrby <- valueY[wtd]                          # training labels
  
  trposflag <- ntrby > 0                      # training labels for diabetes positive
  ptregs <- ntrbx[trposflag, ]                # training rows features with diabetes positive
  ntregs <- ntrbx[!trposflag, ]               # training rows features with diabetes negative
  
  ntebx <- nbx[-wtd, ]                        # test rows - features
  nteby <- valueY[-wtd]                         # test rows - labels
  
  ptrmean <- sapply(ptregs, mean, na.rm = T)  # vector of means for training, diabetes positive
  ntrmean <- sapply(ntregs, mean, na.rm = T)  # vector of means for training, diabetes negative
  ptrsd   <- sapply(ptregs, sd, na.rm = T)    # vector of sd for training, diabetes positive
  ntrsd   <- sapply(ntregs, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
  ptroffsets <- t(t(ntrbx) - ptrmean)         # first step normalize training diabetes pos, subtract mean
  ptrscales  <- t(t(ptroffsets) / ptrsd)      # second step normalize training diabetes pos, divide by sd
  ptrlogs    <- -(1/2) * rowSums(apply(ptrscales, c(1,2),
                function(x) x^2), na.rm = T) - sum(log(ptrsd))+log(NROW(ptregs)/NROW(ntrby))  # Log likelihoods based on 
								# normal distr. for diabetes positive
  ntroffsets <- t(t(ntrbx) - ntrmean)
  ntrscales  <- t(t(ntroffsets) / ntrsd)
  ntrlogs    <- -(1/2) * rowSums(apply(ntrscales, c(1,2) 
                                       , function(x) x^2), na.rm = T) - sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
                                                                # Log likelihoods based on 
								# normal distr for diabetes negative
                                                                # (It is done separately on each class)
  
  lvwtr      <- ptrlogs > ntrlogs              # Rows classified as diabetes positive by classifier 
  gotrighttr <- lvwtr == ntrby                 # compare with true labels
  trscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
  pteoffsets <- t(t(ntebx)-ptrmean)            # Normalize test dataset with parameters from training
  ptescales  <- t(t(pteoffsets)/ptrsd)
  ptelogs    <- -(1/2)*rowSums(apply(ptescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) +log(NROW(ptregs)/NROW(ntrby))
  
  nteoffsets <- t(t(ntebx)-ntrmean)            # Normalize again for diabetes negative class
  ntescales  <- t(t(nteoffsets)/ntrsd)
  ntelogs    <- -(1/2)*rowSums(apply(ntescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
  
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==nteby
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
}
print(tescore)
print(mean(tescore))

#########
#Part B
########

wdat2 <-wdat
#replace 0 to NA
wdat2$V3[wdat2$V3 == 0] <- NA
wdat2$V4[wdat2$V4 == 0] <- NA
wdat2$V6[wdat2$V6 == 0] <- NA
wdat2$V8[wdat2$V8 == 0] <- NA

valueX<-wdat2[,-c(9)]
valueY<-wdat2[,9]
trscore <- array(dim = 10)
tescore <- array(dim = 10)


for (wi in 1:10) {
  wtd <- createDataPartition(y = valueY, p = 0.8, list = FALSE) # 80% of the data into training
  nbx <- valueX                                 # matrix of features
  ntrbx <- nbx[wtd, ]                         # training features
  ntrby <- valueY[wtd]                          # training labels
  
  trposflag <- ntrby > 0                      # training labels for diabetes positive
  ptregs <- ntrbx[trposflag, ]                # training rows features with diabetes positive
  ntregs <- ntrbx[!trposflag, ]               # training rows features with diabetes negative
  
  ntebx <- nbx[-wtd, ]                        # test rows - features
  nteby <- valueY[-wtd]                         # test rows - labels
  
  ptrmean <- sapply(ptregs, mean, na.rm = T)  # vector of means for training, diabetes positive
  ntrmean <- sapply(ntregs, mean, na.rm = T)  # vector of means for training, diabetes negative
  ptrsd   <- sapply(ptregs, sd, na.rm = T)    # vector of sd for training, diabetes positive
  ntrsd   <- sapply(ntregs, sd, na.rm = T)    # vector of sd for training, diabetes negative
  
  ptroffsets <- t(t(ntrbx) - ptrmean)         # first step normalize training diabetes pos, subtract mean
  ptrscales  <- t(t(ptroffsets) / ptrsd)      # second step normalize training diabetes pos, divide by sd
  ptrlogs    <- -(1/2) * rowSums(apply(ptrscales, c(1,2),
                function(x) x^2), na.rm = T) - sum(log(ptrsd))+log(NROW(ptregs)/NROW(ntrby))  # Log likelihoods based on 
								# normal distr. for diabetes positive
  ntroffsets <- t(t(ntrbx) - ntrmean)
  ntrscales  <- t(t(ntroffsets) / ntrsd)
  ntrlogs    <- -(1/2) * rowSums(apply(ntrscales, c(1,2) 
                                       , function(x) x^2), na.rm = T) - sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
                                                                # Log likelihoods based on 
								# normal distr for diabetes negative
                                                                # (It is done separately on each class)
  
  lvwtr      <- ptrlogs > ntrlogs              # Rows classified as diabetes positive by classifier 
  gotrighttr <- lvwtr == ntrby                 # compare with true labels
  trscore[wi]<- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # Accuracy with training set
  
  pteoffsets <- t(t(ntebx)-ptrmean)            # Normalize test dataset with parameters from training
  ptescales  <- t(t(pteoffsets)/ptrsd)
  ptelogs    <- -(1/2)*rowSums(apply(ptescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) +log(NROW(ptregs)/NROW(ntrby))
  
  nteoffsets <- t(t(ntebx)-ntrmean)            # Normalize again for diabetes negative class
  ntescales  <- t(t(nteoffsets)/ntrsd)
  ntelogs    <- -(1/2)*rowSums(apply(ntescales,c(1, 2)
                                     , function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) +log(NROW(ntregs)/NROW(ntrby))
  
  lvwte<-ptelogs>ntelogs
  gotright<-lvwte==nteby
  tescore[wi]<-sum(gotright)/(sum(gotright)+sum(!gotright))  # Accuracy on the test set
}
print(tescore)
print(mean(tescore))



