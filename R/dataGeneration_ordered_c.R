############################################################
############################################################
############################################################
### Function to generate data from mediation scenario
### with two ordered (sequential) mediators
### Include with package ieTest
### Version 2 (version 1 in mmed)
### Not genetic - normally distributed independent variable
############################################################
############################################################
############################################################


#### Function is based on G -> S -> Y for indirect effect
#### Variable names use the above formulation
require(MASS)
dataGen_ordered_c = function( n,                   ### sample size
                    GtoS1 = .15,          ### independent to mediator1
                    S1toS2 = .25,         ### mediator1 to mediator2
                    S2toY = .25,          ### mediator2 to response

                    GS1intS2 = 0,           ### interaction of independent and mediator 1 on mediator 2
                    GS1intY = 0,           ### interaction of independent and mediator 1 on response
                    GS2intY = 0,           ### interaction of independent and mediator 2 on response
                    S1S2int = 0,          ### interaction of both mediators on response
                    GS1S2int = 0,         ### interaction of independent variable and both mediators on response

                    S1toY = .1,           ### First mediator to response
                    GtoS2 = .1,           ### independent to second mediator - nuisance 
                    GtoY  = .2,          ### independent to response - direct effect

                    int1 = 1,           ### Intercept for G to S1
                    int2 = 1,           ### Intercept for S1 to S2
                    int3 = 1,           ### Intercept for S to Y

                    errorMed1Mean = 0,      ### mean of error for G to S1
                    errorMed1SD = sqrt(1),  ### sd of error for G to S1
                    errorMed2Mean = 0,      ### mean of error for S1 to S2
                    errorMed2SD = sqrt(1),  ### sd of error for S1 to S2
                    errorRespMean = 0,      ### mean of error for S2 to Y
                    errorRespSD = sqrt(1),  ### sd of error for S2 to Y

                    probG = c(.5, .5),   ### Probability that G = 0, 1, 2 - length of probability determines possible levels of G
                                         ### If value is null, then will use generated other value to simulate gene and
                                         ### population structure of maf

                    ######## Covariate Options
                    ### 3 covariates: Normal (ethnicity), binary (gender), uniform (age) for each mediator
                    meanEth = 0,
                    sdEth = 1,
                    pGender = .5,         ### Limit to 2 - calculate prob of other in program
                    lowAge = 0,
                    highAge = 1,         ### upper and lower limits of uniform distribution - probably keep at 0, 1

                    ###### Coefficients for mediator and response
                    coef_med1_eth = .5,
                    coef_med1_gender = .5,
                    coef_med1_age = .5,

                    coef_med2_eth = .5,
                    coef_med2_gender = -.5,          #### Negative here by default to show different influence
                    coef_med2_age = .5,

                    coef_resp_eth = .5,
                    coef_resp_gender = .5,
                    coef_resp_age = -.5,

                    #### Missing Data portion - only mediator data can be missing
                    missPer1 = 0,
                    missPer2 = 0,
                    ##### Value for scenario for missing data
                    ## Scen 1 for MCAR (randomly missing)
                    ## 2 - MNAR - higher probability of missing with higher mediator value
                    ## 3 - MAR - Larger response, more likely to be missing
                    ## 4 - MAR - Median response more likely to be missing
                    ## Update March 31, 2021 - Adding non-independent missing sampling
                    ## 5 - 8: 100% dependent - if one mediator is missing, the other will be as well
                    ##        Follow the same pattern as 1-4 (5 - MCAR, 6 - MNAR, 7 - MAR large, 8 - MAR med)
                    scenario = 1,
                    ###### Mean and sd for independent variable
                    indMean = 10, 
                    indSD = 2){
  #### Set missing base value for Scenario 4 - must be larger than missPer
  missBase1 = 1 - ((1 - missPer1) / 3)
  missBase2 = 1 - ((1 - missPer2) / 3)
  #### Generate additional covariates
  ethVar = rnorm(n=n, mean = meanEth, sd = sdEth)
  genderVar = sample(x = c(0, 1), size = n, replace = TRUE, prob = c(pGender, 1 - pGender))
  ageVar = runif(n = n, min = lowAge, max = highAge)

  G = rnorm(n = n, mean = indMean, sd = indSD)
  # while(length(table(G)) == 1){ # loop to ensure matrix is not un-invertable
  #   if(!is.null(probG)){
  #     G = sample(0:(length(probG) - 1), size = n, replace = TRUE, prob = probG)
  #   }else{
  #     maf = exp(0.5 * ethVar) / (1 + exp(0.5*ethVar))
  #     G = rep(0, n)
  #     for(genoI in seq_len(n)){
  #       G[genoI] = sample(0:2, 1, replace = TRUE, prob = c(maf[genoI]^2, 2*maf[genoI]*(1-maf[genoI]), (1 - maf[genoI])^2))
  #     }
  #   }
  # }

  ### Set up X1, X2, and Z
  X1 = X2 = Z = as.matrix(cbind(1, ethVar, genderVar, ageVar))

  ### Generate full S and Y

  fullS1 = as.numeric(X1 %*% c(int1, coef_med1_eth, coef_med1_gender, coef_med1_age) + 
                            G*GtoS1 + rnorm(n, mean = errorMed1Mean, sd = errorMed1SD))
  fullS2 = as.numeric(X2 %*% c(int2, coef_med2_eth, coef_med2_gender, coef_med2_age) + 
                                G*GtoS2 + fullS1*S1toS2 + G*fullS1*GS1intS2 +
                                rnorm(n, mean = errorMed1Mean, sd = errorMed1SD))
  
  Y = as.numeric(Z %*% c(int3, coef_resp_eth, coef_resp_gender, coef_resp_age) +
                   G*GtoY + fullS1*S1toY + fullS2*S2toY + G*fullS1*GS1intY + G*fullS2*GS2intY +
                   fullS1*fullS2*S1S2int + G*fullS1*fullS2*GS1S2int + 
                   rnorm(n = n, mean = errorRespMean, sd = errorRespSD))

  S1 = fullS1
  S2 = fullS2

  toMiss1 = rep(0, n)
  toMiss2 = rep(0, n)
  #### Generate missing Data
  
  ## Each mediator missing independently
  if(scenario %in% 1:4){
    if(missPer1 > 0){
  
      ### Scenario 1 - MCAR - equal probability of being missing
      if(scenario == 1){
        toMiss1 = (1:n) %in% sample(1:n, size = floor(n*missPer1))
      }
      #### Scenario 2 - MNAR - missing dependent on S - higher S, higher missing prob
      if(scenario == 2){ # Missingness dependent upon covariate
        pdS = rank(S1) / n
        if(missPer1 <= .5){
          t1 = missPer1*pdS*2
          sMiss = punif(t1)
        }else{
          t1 = (1-missPer1)*pdS*2
          sMiss = 1 - punif(t1)
        }
        ### Then actually sample using that probability
        for(i in 1:n){
          toMiss1[i] = sample(c(TRUE, FALSE), 1,
                             prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 3 MAR - Larger Y means larger probability of missing
      if(scenario == 3){
        respPercentile = rank(Y)/(n + 1)
        t1 = qnorm(respPercentile)
        t2 = t1 + qnorm(missPer1) + (missPer1 - .5)*1.11205
        sMiss = pnorm(t2)
        for(i in 1:n){
          toMiss1[i] = sample(c(TRUE, FALSE), 1,
                             prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 4 - MAR - Median Y means larger probability of missing
      if(scenario == 4){
        tY = abs(Y - median(Y))
        respOrder = order(tY)
        for(i in respOrder){
          toMiss1[i] = sample(c(TRUE, FALSE), 1, prob = c(missBase1, 1-missBase1))
          if(sum(toMiss1) >= (n*missPer1)){break}                                ### Stop when enough missing
        }
      }
    } # end missing S1 calculation
    ### S2 missingness calculation
    if(missPer2 > 0){
  
      ### Scenario 1 - MCAR - equal probability of being missing
      if(scenario == 1){
        toMiss2 = (1:n) %in% sample(1:n, size = floor(n*missPer2))
      }
      #### Scenario 2 - MNAR - missing dependent on S - higher S, higher missing prob
      if(scenario == 2){ # Missingness dependent upon covariate
        pdS = rank(S2) / n
        if(missPer2 <= .5){
          t1 = missPer2*pdS*2
          sMiss = punif(t1)
        }else{
          t1 = (1-missPer2)*pdS*2
          sMiss = 1 - punif(t1)
        }
        ### Then actually sample using that probability
        for(i in 1:n){
          toMiss2[i] = sample(c(TRUE, FALSE), 1,
                              prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 3 MAR - Larger Y means larger probability of missing
      if(scenario == 3){
        respPercentile = rank(Y)/(n + 1)
        t1 = qnorm(respPercentile)
        t2 = t1 + qnorm(missPer2) + (missPer2 - .5)*1.11205
        sMiss = pnorm(t2)
        for(i in 1:n){
          toMiss2[i] = sample(c(TRUE, FALSE), 1,
                              prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 4 - MAR - Median Y means larger probability of missing
      if(scenario == 4){
        tY = abs(Y - median(Y))
        respOrder = order(tY)
        for(i in respOrder){
          toMiss2[i] = sample(c(TRUE, FALSE), 1, prob = c(missBase2, 1-missBase2))
          if(sum(toMiss2) >= (n*missPer2)){break}                                ### Stop when enough missing
        }
      }
    } # end missing S2 calculation, end indpependent missing simulation
  }else if(scenario %in% 5:8){
    missPer = max(missPer1, missPer2)
    if(missPer > 0){
      if(scenario == 5){
        toMiss1 = (1:n) %in% sample(1:n, size = floor(n*missPer))
      }
      #### Scenario 6 - MNAR - missing dependent on S - higher S, higher missing prob
      if(scenario == 6){ # Missingness dependent upon sum of the mediators
        pdS = rank(S1+S2) / n
        if(missPer <= .5){
          t1 = missPer*pdS*2
          sMiss = punif(t1)
        }else{
          t1 = (1-missPer)*pdS*2
          sMiss = 1 - punif(t1)
        }
        ### Then actually sample using that probability
        for(i in 1:n){
          toMiss1[i] = sample(c(TRUE, FALSE), 1,
                              prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 7 MAR - Larger Y means larger probability of missing
      if(scenario == 7){
        respPercentile = rank(Y)/(n + 1)
        t1 = qnorm(respPercentile)
        t2 = t1 + qnorm(missPer) + (missPer - .5)*1.11205
        sMiss = pnorm(t2)
        for(i in 1:n){
          toMiss1[i] = sample(c(TRUE, FALSE), 1,
                              prob = c(sMiss[i], 1 - sMiss[i]))
        }
      }
      ##### Scenario 8 - MAR - Median Y means larger probability of missing
      if(scenario == 8){
        tY = abs(Y - median(Y))
        respOrder = order(tY)
        for(i in respOrder){
          toMiss1[i] = sample(c(TRUE, FALSE), 1, prob = c(missBase1, 1-missBase1))
          if(sum(toMiss1) >= (n*missPer)){break}                                ### Stop when enough missing
        }
      }
      toMiss2 = toMiss1
    } # if loop for missing percentage
  }### End dependent missingness simulation
  
  R1 = (!toMiss1)*1
  S1[R1 != 1] = NA
  R2 = (!toMiss2)*1
  S2[R2 != 1] = NA

  return(list(Y = Y, S1 = S1, R1 = R1, fullS1 = fullS1,
              S2 = S2, R2 = R2, fullS2 = fullS2,
              G = G, Z = Z, X1 = X1, X2 = X2))
}
