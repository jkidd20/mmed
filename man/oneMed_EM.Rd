\name{oneMed_EM}
\alias{oneMed_EM}
\docType{package}
\title{
EM Algorithm for mediation analysis with a single mediator
}
\description{
Takes response, independent, and mediator values as well as additional covariates
and returns estimates for the effects/coefficients for a mediation model.
}
\usage{
oneMed_EM(Y, G, S, R, lowerS, upperS, Z, X, interaction = TRUE,
               convLimit = 1e-4, iterationLimit = 1e4,
               MEASURE_TYPE_KNOWN = 1,
               MEASURE_TYPE_MISSING = 0,
               MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
               MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
}
\arguments{
  \item{Y}{Numeric vector containing the continuous response variable}
  \item{G}{Numeric vector of the independent variable}
  \item{S}{Numeric vector of the mediating variable. Missing values are allowed}
  \item{R}{Integer vector specifying the status of the mediating variable. Code 
            for these values specified below.}
  
  \item{lowerS}{Numeric vector specifying the smallest value of S that can be observed.
                These values can vary for each measurment}
  \item{upperS}{Numeric vector specifying the largest value of S that can be observed.
                These values can vary for each measurment}
  \item{Z}{Matrix continaing the coefficients to be used for the model for Y}
  \item{X}{Matrix continaing the coefficients to be used for the model for S}
  \item{interaction}{TRUE or FALSE specifying if the interaction term between 
                      G and S should be included}
  \item{convLimit}{Convergence limit of the EM algorithm}
  \item{iteration Limit}{Maximimum number of iterations for the EM algorithm to converge}
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
