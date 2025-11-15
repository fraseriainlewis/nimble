# python driver

library(reticulate)
setwd("~/rstan_nimble_proj/nimble/rtfp/R")
py_run_file("working_joint.py")

a<-py$samples[[5]]; # samples indexes 1-no. params. this is 5th parameter all chains
chain1<-rep(NA,length(a));
chain2<-rep(NA,length(a));
chain3<-rep(NA,length(a));
chain4<-rep(NA,length(a));

for(i in 0:(length(a)-1)){ # 0-indexing
  chain1[i]<-a[[i]]$numpy()[1]
  chain2[i]<-a[[i]]$numpy()[2]
  chain3[i]<-a[[i]]$numpy()[3]
  chain4[i]<-a[[i]]$numpy()[4]
}

plot(chain1,type="l")
lines(chain2,col="red")
lines(chain3,col="magenta")
lines(chain4,col="orange")

