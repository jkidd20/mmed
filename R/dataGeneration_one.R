############################################################
############################################################
############################################################
### Function to generate data from mediation scenario
### with a single mediator.
### For simulations to test performance
### of the EM algorithm approach
### Include with package
### Consider as genetic in nature, but can be set to
### any "additive" data generation scenario
############################################################
############################################################
############################################################


#### Function is based on G -> S -> Y for indirect effect
#### Variable names use the above formulation

dataGen_one = function( n,                   ### sample size
                    GtoS = .15,          ### independent to mediator
                    StoY = .25,          ### mediator to response
                    GSint = 0,           ### interaction of independent and mediator on response
                    GtoY  = .2,          ### independent to response - direct effect
                    int1 = 0,            ### Intercept for G to S
                    int2 = 0,            ### Intercept for S to Y
                    errorMedMean = 0,      ### mean of error for G to S
                    errorMedSD = sqrt(1),  ### sd of error for G to S
                    errorRespMean = 0,      ### mean of error for S to Y
                    errorRespSD = sqrt(1),  ### sd of error for S to Y

                    probG = c(.5, .5),   ### Probability that G = 0, 1, 2 - length of probability determines possible levels of G
                                         ### If value is null, then will use generated other value to simulate gene and
                                         ### population structure of maf

                    ######## Covariate Options
                    ### 3 covariates: Normal (ethnicity), binary (gender), uniform (age)
                    meanEth = 0,
                    sdEth = 1,
                    pGender = .5,         ### Limit to 2 - calculate prob of other in program
                    lowAge = 0,
                    highAge = 1,         ### upper and lower limits of uniform distribution - probably keep at 0, 1

                    ###### Coefficients for mediator and response
                    coef_med_eth = .5,
                    coef_med_gender = .5,
                    coef_med_age = .5,
                    coef_resp_eth = .5,
                    coef_resp_gender = .5,
                    coef_resp_age = .5,

                    #### Missing Data portion - only mediator data can be missing
                    missPer = 0,
                    ##### Value for scenario for missing data
                    ## Scen 1 for MCAR (randomly missing)
                    ## 2 - MNAR - higher probability of missing with higher mediator value
                    ## 3 - MAR - Larger response, more likely to be missing
                    ## 4 - MAR - Median response more likely to be missing
                    scenario = 1){
  #### Set missing base value for Scenario 4 - must be larger than missPer
  missBase = 1 - ((1 - missPer) / 3)
  #### Generate additional covariates
  ethVar = rnorm(n=n, mean = meanEth, sd = sdEth)
  genderVar = sample(x = c(0, 1), size = n, replace = TRUE, prob = c(pGender, 1 - pGender))
  ageVar = runif(n = n, min = lowAge, max = highAge)

  G = 0
  while(length(table(G)) == 1){ # loop to ensure matrix is not un-invertable
    if(!is.null(probG)){
      G = sample(0:(length(probG) - 1), size = n, replace = TRUE, prob = probG)
    }else{
      maf = exp(0.5 * ethVar) / (1 + exp(0.5*ethVar))
      G = rep(0, n)
      for(genoI in seq_len(n)){
        G[genoI] = sample(0:2, 1, replace = TRUE, prob = c(maf[genoI]^2, 2*maf[genoI]*(1-maf[genoI]), (1 - maf[genoI])^2))
      }
    }
  }

  ### Set up X and Z
  X = Z = as.matrix(cbind(1, ethVar, genderVar, ageVar))

  ### Generate full S and Y
  fullS = as.numeric(X %*% c(int1, coef_med_eth, coef_med_gender, coef_med_age) + G*GtoS +
                              rnorm(n = n, mean = errorMedMean, sd = errorMedSD))   ### FullS holds values, S will have missing
  S = fullS
  #browser()
  Y = as.numeric(Z %*% c(int2, coef_resp_eth, coef_resp_gender, coef_resp_age) + G*GtoY + S*StoY +
                  G*S*GSint + rnorm(n = n, mean = errorRespMean, sd = errorRespSD))
  toMiss = rep(0, n)
  #### Generate missing Data
  if(missPer > 0){

    ### Scenario 1 - MCAR - equal probability of being missing
    if(scenario == 1){
      toMiss = (1:n) %in% sample(1:n, size = floor(n*missPer))
    }

    #### Scenario 2 - MNAR - missing dependent on S - higher S, higher missing prob
    if(scenario == 2){ # Missingness dependent upon covariate
      pdS = rank(S) / n
      if(missPer <= .5){
        t1 = missPer*pdS*2
        sMiss = punif(t1)
      }else{
        t1 = (1-missPer)*pdS*2
        sMiss = 1 - punif(t1)
      }
      ### Then actually sample using that probability
      for(i in 1:n){
        toMiss[i] = sample(c(TRUE, FALSE), 1,
                           prob = c(sMiss[i], 1 - sMiss[i]))
      }
    }

    ##### Scenario 3 MAR - Larger Y means larger probability of missing
    if(scenario == 3){
      respPercentile = rank(Y)/(n + 1)
      t1 = qnorm(respPercentile)
      t2 = t1 + qnorm(missPer) + (missPer - .5)*1.11205
      sMiss = pnorm(t2)
      for(i in 1:n){
        toMiss[i] = sample(c(TRUE, FALSE), 1,
                           prob = c(sMiss[i], 1 - sMiss[i]))
      }
    }

    ##### Scenario 4 - MAR - Median Y means larger probability of missing
    if(scenario == 4){
      tY = abs(Y - median(Y))
      respOrder = order(tY)
      for(i in respOrder){
        toMiss[i] = sample(c(TRUE, FALSE), 1, prob = c(missBase, 1-missBase))
        if(sum(toMiss) >= (n*missPer)){break}                                ### Stop when enough missing
      }
    }

  } # end missing calculation

  R = (!toMiss)*1
  S[R != 1] = NA

  return(list(Y = Y, S = S, R = R, fullS = fullS, G = G, Z = Z, X = X))
}
