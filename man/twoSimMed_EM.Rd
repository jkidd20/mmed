\name{twoSimMed_EM}
\alias{twoSimMed_EM}
\docType{package}
\title{
EM Algorithm for mediation analysis with two simultaneous mediators
}
\description{
Takes response, independent, and mediator values as well as additional covariates
and returns estimates for the effects/coefficients for a mediation model.
}
\usage{
twoSeqMed_EM(Y, G, S1, R1, S2, R2, Z, X, 
                lowerS1, upperS1, lowerS2, upperS2, 
                int_gs1 = TRUE, int_gs2 = TRUE, 
                int_s1s2 = FALSE, int_gs1s2 = FALSE,
               convLimit = 1e-4, iterationLimit = 1e4, nDivisions = 5, 
               MEASURE_TYPE_KNOWN = 1,
               MEASURE_TYPE_MISSING = 0,
               MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
               MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
}
\arguments{
  \item{Y}{Numeric vector containing the continuous response variable}
  \item{G}{Numeric vector of the independent variable}
  \item{S1}{Numeric vector of the first mediating variable. Missing values are allowed}
  \item{R1}{Integer vector specifying the status of the first mediating variable. Code 
            for these values specified below.}
  \item{S2}{Numeric vector of the second mediating variable. Missing values are allowed}
  \item{R2}{Integer vector specifying the status of the second mediating variable. Code 
            for these values specified below.}
  \item{Z}{Matrix continaing the coefficients to be used for the model for Y}
  \item{X}{Matrix continaing the coefficients to be used for the model for the mediators}
  \item{lowerS1}{Numeric vector specifying the smallest value of S1 that can be observed.
                These values can vary for each measurment}
  \item{upperS1}{Numeric vector specifying the largest value of S1 that can be observed.
                These values can vary for each measurment}
  \item{lowerS2}{Numeric vector specifying the smallest value of S2 that can be observed.
                These values can vary for each measurment}
  \item{upperS2}{Numeric vector specifying the largest value of S2 that can be observed.
                These values can vary for each measurment}
  \item{int_gs1}{TRUE or FALSE specifying if the interaction term between 
                      G and S1 should be included for the model for S2}
  \item{int_gs2}{TRUE or FALSE specifying if the interaction term between 
                      G and S2 should be included for the model for Y}                   
  \item{int_s1s2}{TRUE or FALSE specifying if the interaction term between 
                      S1 and S2 should be included for the model for Y}
  \item{int_gs1s2}{TRUE or FALSE specifying if the interaction term between 
                      G, S1, and S2 should be included for the model for Y}
  \item{convLimit}{Convergence limit of the EM algorithm}
  \item{iteration Limit}{Maximimum number of iterations for the EM algorithm to converge}
  \item{nDivisions}{Number of initial divisions to be used for numeric integration}
  \item{MEASURE_TYPE_KNOWN}{Integer specifying the key in R for a measured value}
  \item{MEASURE_TYPE_MISSING}{Integer specifying the key in R for a purely missing value}
  \item{MEASURE_TYPE_BELOW_DETECTION_LIMIT}{Integer specifying the key in R for a 
                                              value below detection limit}
  \item{MEASURE_TYPE_ABOVE_DETECTION_LIMIT}{Integer specifying the key in R for a 
                                              value above detection limit}
}
\value{A list containing estimates of the effects in the model}
\references{
}

\examples{
\dontrun{

}
}
