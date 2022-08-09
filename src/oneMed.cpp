#include <RcppDist.h>
#include <RcppNumerical.h>
#include <boost/math/special_functions/bessel.hpp>
#include <RcppArmadillo.h>
#include <math.h>
#include <Rcpp.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppDist)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(BH)]]

using namespace Eigen;
using namespace Rcpp;
using namespace Numer;

/// a value for expected values
//[[Rcpp::export]]
double calc_a_val(const double & gamma, const double & sig2, const double & tau2,
                  const double & psi = 0, const double & G = 0){
  return 1.0 / ((std::pow(gamma + psi*G, 2) / (sig2)) + (1 / (tau2)));
}

// B value for expected values
//[[Rcpp::export]]
double calc_b_val(const double & a, double const & gamma, double const & sig2, double const & tau2,
                  const double & y, const double & G, const arma::rowvec & Z, const arma::rowvec & X,
                  const arma::vec & beta, const arma::vec & ksi,
                  const double & psi = 0){
  return a*( (as_scalar(X*ksi) / tau2) + ((gamma + psi*G)*(y - as_scalar(Z*beta)) / sig2) );
}



// calc_expectation takes the current estimates of the parameters and the
// data as arguments. Additionally takes by reference two vectors for the
// expectation and expectation^2 to fill in.
//[[Rcpp::export]]
List calc_expectation(const arma::vec & beta, const double & gamma, const double & psi, const double & sig2,
                      const arma::vec & ksi, const double & tau2,
                      const arma::vec & Y, const arma::vec & G, const arma::mat & Z, const arma::mat & X,
                      const arma::vec & S, const arma::vec & R,
                      const arma::vec & LLD,
                      const arma::vec & ULD,
                      const int MEASURE_TYPE_KNOWN = 1,
                      const int MEASURE_TYPE_MISSING = 0,
                      const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                      const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
{

  //Rcout << "Started\n";
  const unsigned int N = Z.n_rows;

  // Assign all values to expectation and expectation squared at beginning.
  // Update those that are missing/low/high;
  arma::vec ES = S;
  arma::vec ES2 = arma::square(S);

  for ( unsigned int index = 0; index < N; ++index )
  {
    /* Calculate a and b values for each subject, but only in missing loop to save time on observed */
    if(R(index) == MEASURE_TYPE_MISSING )
    {
      const double a_val = calc_a_val( gamma, sig2, tau2, psi, G(index));
      const double b_val = calc_b_val(a_val, gamma, sig2, tau2,
                                      Y(index),G(index), Z.row(index), X.row(index),
                                      beta, ksi, psi);
      ES(index) = b_val ;
      ES2(index) = b_val*b_val + a_val ;
    }
    else
      if (R(index) == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
      {
        const double a_val = calc_a_val( gamma, sig2, tau2, psi, G(index));
        const double sqrt_a_val = std::sqrt( a_val );
        const double b_val = calc_b_val(a_val, gamma, sig2, tau2,
                                        Y(index), G(index), Z.row(index), X.row(index),
                                        beta, ksi, psi);

        const double arg_val = (LLD(index) - b_val) / sqrt_a_val;
        const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
        const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
        const double func_ratio = pdf_val / cdf_val;
        ES(index) = b_val - sqrt_a_val * func_ratio;

        const double first_term = ES(index)*ES(index);
        const double second_term = a_val *
          (1 - (arg_val*func_ratio) - func_ratio*func_ratio );
        ES2(index) = first_term + second_term;
      }
      else
        if (R(index) == MEASURE_TYPE_ABOVE_DETECTION_LIMIT)
        {
          const double a_val = calc_a_val( gamma, sig2, tau2, psi, G(index));
          const double sqrt_a_val = std::sqrt( a_val );
          const double b_val = calc_b_val(a_val, gamma, sig2, tau2,
                                          Y(index), G(index), Z.row(index), X.row(index),
                                          beta, ksi, psi);

          const double arg_val = ( b_val - ULD(index) ) / sqrt_a_val;
          const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
          const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
          const double func_ratio = pdf_val / cdf_val;

          ES(index) = b_val + sqrt_a_val * func_ratio;

          const double first_term = ES(index)*ES(index);
          const double second_term = a_val *
            (1 - (arg_val*func_ratio) - func_ratio*func_ratio );
          ES2(index) = first_term + second_term;
        }
  } // outer for loop
  return List::create(Named("expect") = ES,
                      Named("expectSq") = ES2);
}



// Use the expected values to calculated the new mle estimates
// of the parameters of interest
//[[Rcpp::export]]
arma::vec calc_beta_gamma_psi(const arma::vec & Y,
                              const arma::mat & Z,   // Was X in previous version - update to Z to match Lin (2020)
                              const arma::vec & G,
                              const arma::vec & ES,
                              const arma::vec & ES2,
                              const bool & interaction = true)
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;
  //Rcout << "Z\n" << Z << "\n\n";
  arma::vec beta_gamma_psi_vec = arma::zeros( N_VAR + 2 ); // + 2 for gamma and psi
  arma::vec Y_expect_vec = arma::zeros( N_VAR + 2 );
  arma::mat Z_expect_mat = arma::zeros( N_VAR + 2, N_VAR + 2);
  for ( unsigned int index = 0; index < N; ++index )
  {
    arma::vec current_vec = arma::zeros( N_VAR + 2 ); // + 2 for gamma and psi
    current_vec.subvec(0, N_VAR - 1) = trans(Y(index)*Z.row(index));  // beta
    current_vec(N_VAR) = Y(index)*ES(index);                          // gamma
    /* If interaction, then calculate */
    if(interaction){
      current_vec(N_VAR+1) = Y(index)*ES(index)*G(index);               //psi
    }
    Y_expect_vec += current_vec;

    arma::mat current_mat = arma::zeros( N_VAR + 2, N_VAR + 2 ); // + 2 for gamma and psi
    current_mat.submat(0, 0, N_VAR - 1, N_VAR - 1) = trans(Z.row(index)) * Z.row(index);           // beta/beta
    current_mat.submat(0, N_VAR, N_VAR-1, N_VAR) = ES(index)*trans(Z.row(index));                  // beta/gamma
    current_mat.submat(N_VAR, 0, N_VAR, N_VAR - 1) = ES(index)*Z.row(index);                       // beta/gamma
    current_mat(N_VAR, N_VAR) = ES2(index);                                                        // gamma/gamma

    /* If interaction, then calculate */
    if(interaction){
      current_mat.submat(0, N_VAR+1, N_VAR-1, N_VAR+1) = ES(index)*trans(Z.row(index))*G(index);     // beta/psi
      current_mat.submat(N_VAR+1, 0, N_VAR+1, N_VAR-1) = ES(index)*Z.row(index)*G(index);            // beta/psi
      current_mat(N_VAR, N_VAR+1) = ES2(index)*G(index);                                             // gamma/psi
      current_mat(N_VAR+1, N_VAR) = ES2(index)*G(index);                                             // gamma/psi
      current_mat(N_VAR+1, N_VAR+1) = ES2(index)*G(index)*G(index);                                  // psi/psi
    }
    Z_expect_mat += current_mat;
  }

  //Rcout << Z_expect_mat << "\n\n";
  //Rcout << Y_expect_vec << "\n\n";
  /* If no interaction, then set to 1 for inverse */
  if(!interaction){
    Z_expect_mat(N_VAR+1, N_VAR+1) = 1;
  }

  arma::mat Z_expect_inv;
  bool invRes = arma::inv(Z_expect_inv, Z_expect_mat);
  if(!invRes){
    Rcout << "Z'Z matrix not invertable\n";
    return arma::zeros(N_VAR + 2);                      // + 2 for gamma and psi
  }
  beta_gamma_psi_vec = Z_expect_inv*Y_expect_vec;
  /* If no interaction, then set psi to 0 */
  if(!interaction){
    beta_gamma_psi_vec(N_VAR+1) = 0;
  }
  return(beta_gamma_psi_vec);
} //calc_beta_gamma_vec


// If no interaction, then psi should be 0, and this will be the same
//[[Rcpp::export]]
double calc_sig2(const double & gamma,
                 const double & psi,
                 const bool & interaction,
                 const arma::vec & beta,
                 const arma::vec & Y,
                 const arma::vec & G,
                 const arma::mat & Z,         // Was X in previous version - update to Z to match Lin (2020)
                 const arma::vec & ES,
                 const arma::vec & ES2 )
{
  const unsigned int N = Z.n_rows;
  const unsigned int N_VAR = Z.n_cols;
  return (1.0/(N - N_VAR - 1 - interaction*1))*accu(pow(Y - Z*beta, 2.0) + pow(gamma + psi*G, 2.0)%ES2 -
          2.0*(gamma + psi*G)%(Y - Z*beta)%ES);
}  //calc_sig2

//calc_ksi : for Maximization step. Use is Alpha in paper, but keeping ksi for coding purposes.
// This function the same for interaction or no interaction
//[[Rcpp::export]]
arma::vec calc_ksi( const arma::mat & X, // Was Z in previous version - update to X to match Lin (2020)
                    const arma::vec & ES)
{
  const unsigned int N = X.n_rows;
  const unsigned int N_VAR = X.n_cols;
  arma::vec expect_X_vec = arma::zeros( N_VAR );
  arma::mat expect_X_mat = arma::zeros( N_VAR, N_VAR );
  for ( unsigned int index = 0; index < N; ++index)
  {
    expect_X_mat += trans(X.row(index)) * X.row(index);
    expect_X_vec += trans(X.row(index)) * ES(index);
  } //for

  arma::mat expect_X_inv;
  bool invRes = arma::inv(expect_X_inv, expect_X_mat);
  if(!invRes){
    Rcout << "X'X matrix not invertable\n";
    arma::vec ksi = arma::zeros(N_VAR);
    ksi(0) = R_NaN;
    return ksi;
  }
  return(expect_X_inv*expect_X_vec);
} //calc_ksi


//calc_tau_sqr : for Maximization step
//[[Rcpp::export]]
double calc_tau2( const arma::vec & ksi, const arma::mat & X,
                  const arma::vec ES,
                  const arma::vec ES2)
{
  const unsigned int N = X.n_rows;
  const unsigned int N_VAR = X.n_cols;
  return (1.0/(N - N_VAR))*accu(ES2 + pow(X*ksi, 2.0) - 2.0*(X*ksi)%ES);
} //calc_tau_sqr



// EM algorithm to iterate between Expectation and Maximization
// Have to input G vector separately as it is needed for interaction
//[[Rcpp::export]]
List oneMed_EM(const arma::vec & Y,
               const arma::vec & G,
               const arma::vec & S,
               const arma::vec & R,
               const arma::vec & lowerS, const arma::vec & upperS,
               const arma::mat & Z,
               const arma::mat & X,
               const bool interaction = true,                         // setting default for interaction to true
               double convLimit = 1e-4, double iterationLimit = 1e4,
               const int MEASURE_TYPE_KNOWN = 1,
               const int MEASURE_TYPE_MISSING = 0,
               const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
               const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){
  int N = Y.n_elem;
  arma::mat Zuse = arma::zeros(N, Z.n_cols + 1);
  arma::mat Xuse = arma::zeros(N, X.n_cols + 1);

  // Update Z and X matrices to add in
  if(Z.n_elem < N){
    Zuse = arma::ones(N, 2);
    Zuse.submat(0, 1, N-1, 1) = G;
  }else{
    Zuse = arma::zeros(N, Z.n_cols + 1);
    //Rcout << Zuse << "\n1\n\n";
    Zuse.submat(0, 0, N-1, 0) = Z.submat(0, 0, N-1, 0);//Rcout << Zuse << "\n2\n\n";
    Zuse.submat(0, 1, N-1, 1) = G;//Rcout << Zuse << "\n3\n\n";
    Zuse.submat(0, 2, N-1, Zuse.n_cols - 1) = Z.submat(0, 1, N-1, Z.n_cols - 1);//Rcout << Zuse << "\n4\n\n";
  }
  if(X.n_elem < N){
    Xuse = arma::ones(N, 2);
    Xuse.submat(0, 1, N-1, 1) = G;
  }else{
    Xuse = arma::zeros(N, X.n_cols + 1);
    Xuse.submat(0, 0, N-1, 0) = X.submat(0, 0, N-1, 0);
    Xuse.submat(0, 1, N-1, 1) = G;
    Xuse.submat(0, 2, N-1, Xuse.n_cols - 1) = X.submat(0, 1, N-1, X.n_cols - 1);
  }

  // Initialize needed values;

  double sig2 = arma::var(Y);
  double tau2 = arma::var(S.elem(find(R == 1)));
  double gamma = 0;
  double psi = 0;
  arma::vec ksi = arma::zeros(Xuse.n_cols);
  arma::vec beta = arma::zeros(Zuse.n_cols);
  bool converged = false;
  int iteration = 0;
  while(!converged & (iteration < iterationLimit)){
    // Create old holders;
    arma::vec oldBeta = beta;
    double oldGamma = gamma;
    double oldPsi = psi;
    double oldSig2 = sig2;
    arma::vec oldKsi = ksi;
    double oldTau2 = tau2;
    //Update Expectation;
    List expRes = calc_expectation(beta, gamma, psi, sig2, ksi, tau2,
                                   Y, G, Zuse, Xuse, S, R,
                                   lowerS, upperS,
                                   MEASURE_TYPE_KNOWN,
                                   MEASURE_TYPE_MISSING,
                                   MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                   MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

    arma::vec expect = expRes["expect"];
    arma::vec expectSq = expRes["expectSq"];

    arma::vec tVector1 = calc_beta_gamma_psi(Y, Zuse, G, expect, expectSq, interaction);

    beta = tVector1.subvec(0, tVector1.n_elem - 3);                                  // -3: 1 for gamma, one for psi, one for offset
    gamma = tVector1(tVector1.n_elem - 2);                                           // -2: 1 for psi, one for offset
    psi = tVector1(tVector1.n_elem - 1);                                             // -1:one for offset
    sig2 = calc_sig2(gamma, psi, interaction, beta, Y, G, Zuse, expect, expectSq);
    ksi = calc_ksi(Xuse, expect);
    tau2 = calc_tau2(ksi, Xuse, expect, expectSq);

    if(std::isnan(gamma) | std::isnan(psi) | std::isnan(ksi(0)) | std::isnan(beta(0))){
      gamma = R_NaN;
      psi = R_NaN;
      sig2 = R_NaN;
      tau2 = R_NaN;
      return List::create(Named("beta") = beta,
                          Named("gamma") = gamma,
                          Named("psi") = gamma,
                          Named("sig2") = sig2,
                          Named("ksi") = ksi,
                          Named("tau2") = tau2);
    }
    // Check for convergence;
    double maxDiff = 1;
    maxDiff = std::max(max(abs(beta - oldBeta)), max(abs(beta - oldBeta) / abs(oldBeta)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma - oldGamma), fabs(gamma - oldGamma) / fabs(oldGamma)));
    maxDiff = std::max(maxDiff, std::max(fabs(psi - oldPsi), fabs(psi - oldPsi) / fabs(oldPsi)));
    maxDiff = std::max(maxDiff, std::max(max(abs(ksi - oldKsi)), max(abs(ksi - oldKsi) / abs(oldKsi))));
    maxDiff = std::max(maxDiff, std::max(fabs(sig2 - oldSig2), fabs(sig2 - oldSig2) / fabs(oldSig2)));
    maxDiff = std::max(maxDiff, std::max(fabs(gamma - oldGamma), fabs(gamma - oldGamma) / fabs(oldGamma)));
    maxDiff = std::max(maxDiff, std::max(fabs(tau2 - oldTau2), fabs(tau2 - oldTau2) / fabs(oldTau2)));

    converged = maxDiff < convLimit;
    iteration++;
  }
  /// If fails to converge, return R_NaN for all single values to be checked in bootstrap;
  if(iteration == iterationLimit){
    Rcout << "Algorithm failed to converge\n";
    return List::create(Named("beta") = beta,
                        Named("gamma") = R_NaN,
                        Named("psi") = R_NaN,
                        Named("sig2") = R_NaN,
                        Named("ksi") = ksi,
                        Named("tau2") = R_NaN);
  }
  return List::create(Named("beta") = beta,
                      Named("gamma") = gamma,
                      Named("psi") = psi,
                      Named("sig2") = sig2,
                      Named("ksi") = ksi,
                      Named("tau2") = tau2);
}


//[[Rcpp::export]]
arma::mat calc_Q_matrix(const double & sig2, const double & tau2,
                        const arma::mat & Z, const arma::mat & X,
                        const arma::vec & G,
                        const arma::vec & ES, const arma::vec & ES2,
                        const bool & interaction = true)
{
  const unsigned int X_VARS = X.n_cols;
  const unsigned int Z_VARS = Z.n_cols;
  const unsigned int N = Z.n_rows;

  //// Formulation of Q Matrix:
  // Beta (intercept, G, covariates), gamma, psi, sigma, ksi, tau
  arma::mat Q = arma::zeros( Z_VARS + 3 + X_VARS + 1, Z_VARS + 3 + X_VARS + 1);


  // Loop for summation elements
  for ( unsigned int index = 0; index < N; ++index )
  {
    Q.submat(0, 0, Z_VARS - 1, Z_VARS - 1) += (trans(Z.row(index)) * Z.row(index));
    Q.submat(0, Z_VARS, Z_VARS-1, Z_VARS) += (ES(index)*trans(Z.row(index))) ;
    Q.submat(Z_VARS, 0, Z_VARS, Z_VARS - 1) += (ES(index)*Z.row(index)) ;
    Q(Z_VARS, Z_VARS) += ES2(index);
    // Calculate Psi values if interaction is true
    if(interaction){
      Q.submat(0, Z_VARS + 1, Z_VARS-1, Z_VARS + 1) += (ES(index)*trans(Z.row(index)) * G(index)) ;
      Q.submat(Z_VARS + 1, 0, Z_VARS + 1, Z_VARS - 1) += (ES(index)*Z.row(index)*G(index)) ;
      Q(Z_VARS, Z_VARS + 1) += ES2(index)*G(index);
      Q(Z_VARS + 1, Z_VARS) += ES2(index)*G(index);
      Q(Z_VARS + 1, Z_VARS + 1) += ES2(index)*G(index)*G(index);
    }
    // +2 here for three parameters minus one for offset
    Q.submat(Z_VARS + 3, Z_VARS + 3, Z_VARS + X_VARS + 2, Z_VARS + X_VARS + 2) += (trans(X.row(index)) * X.row(index));
  }

  Q.submat(0, 0, Z_VARS + 1, Z_VARS + 1) /= sig2;
  Q.submat(Z_VARS + 3, Z_VARS + 3, Z_VARS + X_VARS + 2, Z_VARS + X_VARS + 2) /= tau2;
  Q(Z_VARS + 2, Z_VARS + 2) = N / (2.0*(sig2*sig2));
  Q(Z_VARS + X_VARS + 3, Z_VARS + X_VARS + 3) = N / (2.0 * (tau2*tau2));

  if(!interaction){
    Q(Z_VARS + 1, Z_VARS + 1) = 1;
  }

  return Q;
}


// calc_V_mat returns V matrix for a single observation
//[[Rcpp::export]]
arma::mat calc_V_mat(const double & gamma, const double & psi,
                     const double & sig2, double tau2,
                     const double & Y, const double & G,
                     const arma::vec & beta, const arma::vec & ksi,
                     const arma::rowvec & Z, const arma::rowvec & X)
{
  const unsigned int X_VARS = beta.n_elem;
  const unsigned int Z_VARS = ksi.n_elem;
  arma::mat V_mat = arma::zeros(Z_VARS + X_VARS + 4, 3); // beta, ksi, gamma, psi, sigma2, tau2.

  const double current_residual = Y - as_scalar(Z*beta);

  //first column = 0
  V_mat.submat( 0, 0, Z_VARS-1, 0 ) = current_residual/sig2 * trans(Z);
  //current_V_mat( Z_VARS, 0 ) = 0.0; // gamma
  //current_V_mat( Z_VARS + 1, 0 ) = 0.0; // psi
  V_mat( Z_VARS + 2, 0 ) = (-0.5 / sig2) +
    0.5 * std::pow(current_residual, 2.0) /
      std::pow(sig2, 2.0);
  V_mat.submat( Z_VARS + 3, 0, Z_VARS + 3 + X_VARS - 1, 0 ) =       // beta, gamma, psi, sigma, calc ksi
    -(1.0/tau2) * as_scalar(X*ksi)*trans(X);
  V_mat( Z_VARS + X_VARS + 2, 0 ) =                               // beta, gamma, psi, sigma, ksi, calc tau
    (-0.5 / tau2 ) +
    (0.5/(std::pow(tau2, 2.0)) *
    std::pow(as_scalar(X*ksi), 2.0));

    //second_column = 1
    V_mat.submat( 0, 1, Z_VARS - 1, 1 ) = -( (gamma + psi*G) /sig2 ) * trans(Z);
    V_mat( Z_VARS, 1 ) = (1.0/sig2) * current_residual;
    V_mat( Z_VARS + 1, 1 ) = G*(1.0/sig2) * current_residual;

    V_mat( Z_VARS + 2, 1 ) = -((gamma + psi*G)/std::pow(sig2, 2.0))*current_residual;
    V_mat.submat( Z_VARS + 3, 1, Z_VARS + 3 + X_VARS - 1, 1 ) = trans(X) / tau2;
    V_mat( Z_VARS + X_VARS + 3, 1 ) = (-1.0 / std::pow(tau2, 2)) * as_scalar(X*ksi);

    //third_column = 2
    V_mat( Z_VARS, 2 ) = -(gamma + psi*G) / sig2;
    V_mat( Z_VARS + 1, 2 ) = -G*(gamma + psi*G) / sig2;
    V_mat( Z_VARS + 2, 2 ) =  (0.5*std::pow(gamma + psi*G, 2.0) ) / std::pow(sig2, 2.0);
    V_mat( Z_VARS + X_VARS + 3, 2 ) = 0.5 / std::pow(tau2, 2.0);

    return V_mat;
} //calc_V_mat_vec


// Calculate the U Matrix for a single observation
// Need to know detection limits and where observation falls
//[[Rcpp::export]]
arma::mat calc_U_mat(const double & gamma, const double & psi, const double & sig2, double tau2,
                     const arma::vec & beta,
                     const arma::vec & ksi,
                     const double & Y, const double & G, const double & R,
                     const arma::rowvec & Z,
                     const arma::rowvec & X,
                     const double & LLD, const double & ULD,
                     const int MEASURE_TYPE_KNOWN = 1,
                     const int MEASURE_TYPE_MISSING = 0,
                     const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                     const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2)
{
  arma::mat U_mat = arma::zeros( 3, 3 );

  if ( R == MEASURE_TYPE_KNOWN)
  {
    return U_mat;
  }
  const double a_val = calc_a_val( gamma, sig2, tau2, psi, G);
  const double b_val = calc_b_val(a_val, gamma, sig2, tau2,
                                  Y, G, Z, X,
                                  beta, ksi, psi);

  if ( R == MEASURE_TYPE_MISSING)
  {
    const double mu_1 = b_val;
    const double mu_2 = b_val * b_val + a_val;
    const double mu_3 = 3*a_val*b_val + pow(b_val, 3);
    const double mu_4 = 3*a_val*a_val + 6*a_val*b_val*b_val + pow(b_val, 4);
    U_mat( 0, 0 ) = 1.0;
    U_mat( 0, 1 ) = mu_1;
    U_mat( 0, 2 ) = mu_2;
    U_mat( 1, 0 ) = mu_1;
    U_mat( 1, 1 ) = mu_2;
    U_mat( 1, 2 ) = mu_3;
    U_mat( 2, 0 ) = mu_2;
    U_mat( 2, 1 ) = mu_3;
    U_mat( 2, 2 ) = mu_4;
  }
  else
    if (  R == MEASURE_TYPE_BELOW_DETECTION_LIMIT)
    {
      const double sqrt_a_val = sqrt( a_val );
      const double arg_val = (LLD - b_val) / sqrt_a_val;
      const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
      const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
      const double func_ratio = pdf_val / cdf_val;

      double f_arr[ 5 ];
      f_arr[ 0 ] = 1.0;
      f_arr[ 1 ] = -func_ratio;
      for ( int j = 2; j < 5; ++j )
      {
        const double second_term = (j-1)*f_arr[j-2];
        const double first_term =
          - ( std::pow(arg_val, j-1)  * func_ratio );
          f_arr[ j ] = first_term + second_term;
      }
      double mu_arr[5];
      //mu_arr[0] is unused
      for ( int k = 1; k < 5; ++k )
      {
        mu_arr[ k ] = 0.0;
        for ( int j = 0; j <= k; ++j )
        {
          const double k_choose_j =
            boost::math::factorial<double>( k ) /
              ( boost::math::factorial<double>( j ) *
                boost::math::factorial<double>( k-j ) );
          const double a_b_term =
            pow( a_val, j/2.0) * pow( b_val, k-j);
          mu_arr[k] += (k_choose_j * a_b_term * f_arr[ j ]);
        } //for j loop
      } //for k loop
      const double mu_1 = mu_arr[1];
      const double mu_2 = mu_arr[2];
      const double mu_3 = mu_arr[3];
      const double mu_4 = mu_arr[4];
      U_mat( 0, 0 ) = 1.0;
      U_mat( 0, 1 ) = mu_1;
      U_mat( 0, 2 ) = mu_2;
      U_mat( 1, 0 ) = mu_1;
      U_mat( 1, 1 ) = mu_2;
      U_mat( 1, 2 ) = mu_3;
      U_mat( 2, 0 ) = mu_2;
      U_mat( 2, 1 ) = mu_3;
      U_mat( 2, 2 ) = mu_4;
    } // MEASURE_TYPE_BELOW_DETECT_LIMIT
    else
      if (  R ==  MEASURE_TYPE_ABOVE_DETECTION_LIMIT )
      {
        const double sqrt_a_val = sqrt( a_val );
        const double arg_val = (b_val - ULD) / sqrt_a_val;
        const double pdf_val = R::dnorm(arg_val, 0.0, 1.0, 0);
        const double cdf_val = R::pnorm(arg_val, 0.0, 1.0, 1, 0);
        const double func_ratio = pdf_val / cdf_val;

        double f_arr[ 5 ];
        f_arr[ 0 ] = 1.0;
        f_arr[ 1 ] = -func_ratio;
        for ( int j = 2; j < 5; ++j )
        {
          const double second_term = (j-1)*f_arr[j-2];
          const double first_term =
            - ( pow(arg_val, j-1)  * func_ratio );
            f_arr[ j ] = first_term + second_term;
        }
        double mu_arr[5];
        //mu_arr[0] is unused
        for ( int k = 1; k < 5; ++k )
        {
          mu_arr[ k ] = 0.0;
          for ( int j = 0; j <= k; ++j )
          {
            const double k_choose_j =
              boost::math::factorial<double>( k ) /
                ( boost::math::factorial<double>( j ) *
                  boost::math::factorial<double>( k-j ) );
            const double a_b_term =
              pow( a_val, j/2.0) * pow( -b_val, k-j);
            mu_arr[k] += (k_choose_j * a_b_term * f_arr[ j ]);
          } //for j loop
          if ( ( k % 2 ) == 1 )
          {
            mu_arr[k] = -mu_arr[k];
          }
        } //for k loop
        const double mu_1 = mu_arr[1];
        const double mu_2 = mu_arr[2];
        const double mu_3 = mu_arr[3];
        const double mu_4 = mu_arr[4];
        U_mat( 0, 0 ) = 1.0;
        U_mat( 0, 1 ) = mu_1;
        U_mat( 0, 2 ) = mu_2;
        U_mat( 1, 0 ) = mu_1;
        U_mat( 1, 1 ) = mu_2;
        U_mat( 1, 2 ) = mu_3;
        U_mat( 2, 0 ) = mu_2;
        U_mat( 2, 1 ) = mu_3;
        U_mat( 2, 2 ) = mu_4;
      } // MEASURE_TYPE_ABOVE_DETECT_LIMIT

      return U_mat;
}


// Functions exist for U matrix and V matrix, each for a single observation;
//[[Rcpp::export]]
arma::mat calc_OMEGA(const arma::vec & Y,
                     const arma::vec & G,
                     const arma::vec & S,
                     const arma::vec & R,
                     const arma::vec & lowerS, const arma::vec & upperS,
                     const arma::vec & beta, const double & gamma, const double & psi, const double & sig2,
                     const arma::vec & ksi, const double & tau2,
                     const arma::mat & Z = 0,
                     const arma::mat & X = 0,
                     const bool & interaction = true,
                     const int MEASURE_TYPE_KNOWN = 1,
                     const int MEASURE_TYPE_MISSING = 0,
                     const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                     const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  const unsigned int N = Y.n_elem;
  arma::mat Zuse;
  arma::mat Xuse;
  // Update Z and X matrices to add in
  if(Z.n_elem < N){
    Zuse = arma::ones(N, 2);
    Zuse.submat(0, 1, N-1, 1) = G;
  }else{
    Zuse = arma::zeros(N, Z.n_cols + 1);
    Zuse.submat(0, 0, N-1, 0) = Z.submat(0, 0, N-1, 0);//Rcout << Zuse << "\n2\n\n";
    Zuse.submat(0, 1, N-1, 1) = G;//Rcout << Zuse << "\n3\n\n";
    Zuse.submat(0, 2, N-1, Zuse.n_cols - 1) = Z.submat(0, 1, N-1, Z.n_cols - 1);//Rcout << Zuse << "\n4\n\n";
  }
  if(X.n_elem < N){
    Xuse = arma::ones(N, 2);
    Xuse.submat(0, 1, N-1, 1) = G;
  }else{
    Xuse = arma::zeros(N, X.n_cols + 1);
    Xuse.submat(0, 0, N-1, 0) = X.submat(0, 0, N-1, 0);
    Xuse.submat(0, 1, N-1, 1) = G;
    Xuse.submat(0, 2, N-1, Xuse.n_cols - 1) = X.submat(0, 1, N-1, X.n_cols - 1);
  }
  const unsigned int Z_VARS = Zuse.n_cols;
  const unsigned int X_VARS = Xuse.n_cols;

  // Update expectation using coverged estimates;
  List tExpect = calc_expectation(beta, gamma, psi, sig2, ksi, tau2,
                                  Y, G, Zuse, Xuse, S, R,
                                  lowerS, upperS,
                                  MEASURE_TYPE_KNOWN,
                                  MEASURE_TYPE_MISSING,
                                  MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                  MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

  arma::vec expect = tExpect["expect"];
  arma::vec expectSq = tExpect["expectSq"];

  // Calculate Q matrix

  arma::mat Q = calc_Q_matrix( sig2, tau2, Zuse, Xuse, G, expect, expectSq, interaction);

  arma::mat Qalt = arma::zeros(Z_VARS + X_VARS + 4, Z_VARS + X_VARS + 4);  // beta, kis, gamma, psi, sigma, tau

  // Go through each subject calculating values and summing matrix;
  for(unsigned int index = 0; index < N; ++index ){
    if( R(index) == MEASURE_TYPE_KNOWN ){ continue; }

    arma::vec uVec = arma::zeros(3);

    uVec(0) = 1;
    uVec(1) = expect(index);
    uVec(2) = expectSq(index);

    arma::mat V = calc_V_mat(gamma, psi, sig2, tau2, Y(index), G(index), beta, ksi, Zuse.row(index), Xuse.row(index));

    arma::mat uMat = calc_U_mat(gamma, psi, sig2, tau2, beta, ksi,
                                Y(index), G(index), R(index),
                                Zuse.row(index), Xuse.row(index),
                                lowerS(index), upperS(index),
                                MEASURE_TYPE_KNOWN,
                                MEASURE_TYPE_MISSING,
                                MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

    Qalt += V * uMat * trans(V) - (V*uVec * trans(V*uVec));
  }

  arma::mat omega = Q - Qalt;
  arma::mat omegaInv;

  /// if interaction, force row and column to only 1 on diagonal
  if(!interaction){
    omega.submat(Z_VARS+1, 0, Z_VARS+1, omega.n_cols - 1) = arma::zeros(1, omega.n_cols);
    omega.submat(0, Z_VARS+1, omega.n_cols - 1, Z_VARS+1) = arma::zeros(omega.n_cols, 1);
    omega(Z_VARS+1,Z_VARS+1) = 1;
  }
 
  bool badSig = inv(omegaInv, omega);
  if(badSig){
    /* if no interaction, force variance to 0 */
    if(!interaction){
      omegaInv(Z_VARS+1,Z_VARS+1) = 0;
    }
    return omegaInv;
  } else{
    Rcout << "Estimated Variance Matrix is Singular\n";
    return arma::zeros(X_VARS + Z_VARS + 3, X_VARS + Z_VARS + 3);
  }
}



////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////  Confidence Intervals //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     Delta Method    //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List deltaCI_one(const long double & mu1, const long double & sig1,
                 const long double & mu2, const long double & sig2, const long double & sig12,
                 const double & indL = 1,
                 const long double & mu3 = 0, const long double & sig3 = 0,
                 const long double & sig13 = 0, const long double & sig23 = 0,
                 const double alpha = 0.05){
  double zval = R::qnorm(1.0 - alpha/2.0, 0, 1, 1, 0);
  double deltaSE = 0;
  double deltaEst = mu1*(mu2 + indL*mu3);

  deltaSE = std::sqrt( std::pow(mu2 + indL*mu3, 2.0)*sig1 +
    std::pow(mu1, 2.0)*(sig2 + std::pow(indL, 2.0)*sig3) +
    2.0*( deltaEst*sig12 +
    std::pow(mu1, 2.0)*indL*sig23 +
    deltaEst*indL*sig13));

  arma::vec CI = arma::zeros(2);
  CI(0) = mu1*(mu2 + indL*mu3) - zval*deltaSE;
  CI(1) = mu1*(mu2 + indL*mu3) + zval*deltaSE;

  long double pval = R::pnorm(fabs(mu1*(mu2 + indL*mu3)) / deltaSE, 0, 1, 0, 0);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     bootstrap app    //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List bootstrapCI_one(const arma::vec & y, const arma::vec & g, const arma::vec & s, const arma::vec & r,
                     const arma::mat & Z, const arma::mat & X,
                     const arma::vec & lowers, const arma::vec & uppers, const double & delta,
                     const double & alpha,
                     const bool & interaction = true,
                     const double & indL = 1,
                     const unsigned int bootStrapN = 1000,
                     double convLimit = 1e-4, const double iterationLimit = 1e4,
                     const int MEASURE_TYPE_KNOWN = 1,
                     const int MEASURE_TYPE_MISSING = 0,
                     const int MEASURE_TYPE_BELOW_DETECTION_LIMIT = -1,
                     const int MEASURE_TYPE_ABOVE_DETECTION_LIMIT = 2){

  unsigned int N = y.n_elem;
  //arma::uvec bootSample(N);
  arma::uvec s1;

  double lowCut = alpha/2.0;
  unsigned int lowCutVal = round(bootStrapN * lowCut) - 1;
  double highCut = 1 - lowCut;
  unsigned int highCutVal = round(bootStrapN * highCut) + 1;

  if(highCutVal >= bootStrapN | lowCutVal <= 0){
    highCutVal = bootStrapN - 1;
    lowCutVal = 0;
    Rcout << "Too few iterations used for accurate Bootstrap CI\n";
  }

  //IntegerVector toSample =  seq(1, N) - 1;

  arma::vec deltaHats = arma::zeros(bootStrapN);
  arma::uvec bootSample(N);
  int badSample  = 0;
  for(int i = 0; i < bootStrapN; i++){
    for(int j = 0; j < N; j++){
      bootSample(j) = R::runif(0,N-1);
    }
    /// Check to see if new X and Z are inevitable. If not, decrease i and start sample over
    arma::mat zBoot = Z.rows(bootSample);
    arma::mat xBoot = X.rows(bootSample);
    arma::vec gBoot = g.elem(bootSample);

    arma::mat zBootCheck = arma::zeros(N, zBoot.n_cols + 1);
    zBootCheck.submat(0, 0, N-1, zBoot.n_cols - 1) = zBoot;
    zBootCheck.submat(0, zBoot.n_cols, N-1, zBoot.n_cols) = gBoot;

    arma::mat xBootCheck = arma::zeros(N, xBoot.n_cols + 1);
    xBootCheck.submat(0, 0, N-1, xBoot.n_cols - 1) = xBoot;
    xBootCheck.submat(0, xBoot.n_cols, N-1, xBoot.n_cols) = gBoot;

    arma::mat t_X_inv;
    arma::mat t_Z_inv;

    bool xCheck = arma::inv(t_X_inv, trans(xBootCheck)*xBootCheck);
    bool zCheck = arma::inv(t_Z_inv, trans(zBootCheck)*zBootCheck);
    if(!xCheck | !zCheck){
      i--;
    }else{

      arma::vec yBoot = y.elem(bootSample);
      arma::vec sBoot = s.elem(bootSample);
      arma::vec rBoot = r.elem(bootSample);
      arma::vec lowBoot = lowers.elem(bootSample);
      arma::vec upBoot = uppers.elem(bootSample);

      List itEM_res = oneMed_EM(yBoot, gBoot, sBoot, rBoot, lowBoot, upBoot,
                                zBoot, xBoot,
                                interaction,
                                convLimit, iterationLimit,
                                MEASURE_TYPE_KNOWN,
                                MEASURE_TYPE_MISSING,
                                MEASURE_TYPE_BELOW_DETECTION_LIMIT,
                                MEASURE_TYPE_ABOVE_DETECTION_LIMIT);

      double tGamma = itEM_res["gamma"];
      double tpsi = itEM_res["psi"];
      arma::vec tKsi = itEM_res["ksi"];

      deltaHats(i) = tKsi(1)*(tGamma + indL*tpsi);

      if(!arma::is_finite(deltaHats(i)) | isnan(deltaHats(i))){
        --i;
        ++badSample;
        // If bad sample/no estimate, redo
      }
      if(badSample >= (bootStrapN*1e3)){
        Rcout << "Too many bootstrap samples giving impossible results. returning with estimates available. Results may be inaccurate\n";
        if(i < 2){
          arma::vec CI = arma::zeros(2);
          CI(0) = -1e15;
          CI(1) = 1e15;
          double deltaSE = 1e15;
          double pval = 1;
          return List::create(Named("CI") = CI,
                              Named("deltaSE") = deltaSE,
                              Named("pval") = pval);
        }
        arma::vec temp2 = sort(deltaHats.subvec(0, i - 1));
        arma::vec CI = arma::zeros(2);
        CI(0) = temp2(std::max(floor(lowCutVal * i / bootStrapN), 0.0));
        CI(1) = temp2(std::min(ceil(highCutVal * i / bootStrapN), i - 1.0));
        double deltaSE = stddev(temp2);

        if(delta < 0){
          s1 = arma::find(temp2 >= 0);
        }else if(delta > 0){
          s1 = arma::find(temp2 <= 0);
        }else if(delta == 0){
          s1 = arma::find(temp2 > -999999999999999999);
        }

        double pval = std::min(1.0, 2.0*s1.n_elem/i);

        return List::create(Named("CI") = CI,
                            Named("deltaSE") = deltaSE,
                            Named("pval") = pval);
      }
    }
  }

  arma::vec temp = sort(deltaHats);
  arma::vec CI = arma::zeros(2);
  CI(0) = temp(lowCutVal);
  CI(1) = temp(highCutVal);
  double deltaSE = stddev(deltaHats);

  if(delta < 0){
    s1 = arma::find(temp >= 0);
  }else if(delta > 0){
    s1 = arma::find(temp <= 0);
  }else if(delta == 0){
    s1 = arma::find(temp > -999999999999999999);
  }

  double pval = std::min(1.0, 2.0*s1.n_elem/bootStrapN);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////     MC Approach    ///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List mcCI_one(const arma::vec & mu, const arma::mat & sig, const double & delta,
              const double & indL = 1,
              const double nIt = 10000, const double alpha = 0.05){

  double outVal = std::ceil((alpha/2.0) * nIt);
  double itGen = outVal + 1;
  double itGen2 = itGen;
  double totGen = 0;
  double pMean, cMean = 0;
  double pVar, cVar = 0;
  double aMean = 0;
  double pval = 0;

  double indLMult = indL;
  arma::mat sigUse = sig;

  if(sig(2, 2) == 0){
    indLMult = 0;
    sigUse(2, 2) = 1.0;
  }
  arma::mat tGen = rmvnorm(itGen2 * 2, mu, sigUse);
  arma::vec genValues1 = tGen.col(0) % (tGen.col(1) + indLMult*tGen.col(2));

  arma::vec low = sort(genValues1);
  arma::vec high = sort(genValues1, "descend");

  pMean = arma::mean(genValues1);
  pVar = arma::var(genValues1);

  if(delta < 0){
    arma::uvec s1 = arma::find(genValues1 >= 0);
    pval += s1.n_elem;
  }else if(delta > 0){
    arma::uvec s1 = arma::find(genValues1 <= 0);
    pval += s1.n_elem;
  }else if(delta == 0){
    pval += genValues1.n_elem;
  }

  totGen = itGen2 * 2;
  while(totGen < nIt){
    itGen2 = std::min(nIt - totGen, itGen);

    arma::mat tGen = rmvnorm(itGen2, mu, sigUse);
    arma::vec genValues = tGen.col(0) % (tGen.col(1) + indLMult*tGen.col(2));

    low.subvec(low.n_elem - genValues.n_elem, low.n_elem - 1) = genValues;
    high.subvec(high.n_elem - genValues.n_elem, high.n_elem - 1) = genValues;

    low = sort(low);
    high = sort(high, "descend");

    cMean = arma::mean(genValues);
    cVar = arma::var(genValues);
    aMean = (cMean*itGen2 + pMean*totGen) / (totGen + itGen2);

    pVar = (((totGen * (pVar + std::pow(pMean, 2))) + (itGen2 * (cVar + std::pow(cMean, 2)))) /
      (totGen + itGen2)) - std::pow(aMean, 2);

    totGen += itGen2;
    pMean = aMean;

    if(delta < 0){
      arma::uvec s1 = arma::find(genValues >= 0);
      pval += s1.n_elem;
    }else if(delta > 0){
      arma::uvec s1 = arma::find(genValues <= 0);
      pval += s1.n_elem;
    }else if(delta == 0){
      pval += genValues.n_elem;
    }
  }

  arma::vec CI = arma::zeros(2);
  CI(0) = low(outVal);
  CI(1) = high(outVal);
  double deltaSE = std::sqrt(pVar * nIt / (nIt-1));
  pval = (2.0*pval) / (1.0 * nIt);

  return List::create(Named("CI") = CI,
                      Named("deltaSE") = deltaSE,
                      Named("pval") = pval);
}

// Not working right now. Don't use yet.
// [[Rcpp::export]]
double mcp_calc_one(const arma::vec mu, const arma::mat sig, const double delta,
                    const double & indL = 1,
                    const double nIt = 10000, const double itPer = 1000){
  double totGen = 0;
  double itGen;
  double pval = 0;

  while(totGen < nIt){
    itGen = std::min(nIt - totGen, itPer);
    arma::mat tGen = rmvnorm(itGen, mu, sig);
    arma::vec genValues = tGen.col(0) % (tGen.col(1) + indL*tGen.col(2));
    genValues = arma::abs(genValues);
    arma::uvec temp = arma::find(genValues >= std::abs(delta));
    pval += temp.n_elem;

    totGen += itGen;
  }
  return pval / nIt;
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////   Does not seem to be working at the current moment: do not use!
/////////////////////Product of Normal RV///////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// [[Rcpp::export]]
long double prodExPDF(long double x, long double mu1_in, long double mu2_in, long double mu3_in,
                      long double sig1_in, long double sig2_in, long double sig3_in,
                      long double sig12_in, long double sig13_in, long double sig23_in,
                      double indL = 1,                                                            //indL denotes the level of the independent variable to use
                      int UPPERLIM = 30, long double maxRet = 1e300){

  //// Now set up for interactions
  // mu1 is for alpha_G, mu2 for gamma, mu3 for psi.
  // If psi is zero, then no interaction and same as before

  /// Set values to mu1 and mu2, sig1, sig2, sig12 as used previously
  long double mu1 = mu1_in;
  long double sig1 = sig1_in;

  long double mu2 = mu2_in + indL*mu3_in;
  long double sig2 = sig2_in + std::pow(indL, 2.0)*sig3_in +
    2*indL*sig23_in;
  long double sig12 = sig12_in + indL*sig13_in;


  /// All after here is the same as before
  long double sig1sig2 = std::sqrt(sig1*sig2);
  long double p = sig12 / sig1sig2;

  if(x == 0){
    long double toReturn = ((long double)std::pow(std::sqrt(sig1), - 1)) /
      (long double)(M_PI * (long double)std::pow(1 - std::pow(p, 2), 0.5)*(long double)std::sqrt(sig2)) * LDBL_MAX;
    if(toReturn > maxRet){return maxRet;}
    return toReturn;
  }

  long double absX = fabs(x);

  long double sumT = 0;
  //double factN = 1;
  double factN2 = 1;
  long double besselV = 0;
  long double p1n1, p1n2, p1n3, p1d1, p1d2, p3, p4;
  for(int n = 0; n <= UPPERLIM; n++){
    if(n > 0){
      //factN *= n;
      factN2 *= 2*n * (2*n-1);
    }
    for(int m = 0; m <= 2*n; m++){
      try{
        besselV = boost::math::cyl_bessel_k((long double)(m - n), (long double)(std::abs(x)/((1 - std::pow(p, 2))*sig1sig2)));
      }
      catch (std::exception& ex){
        besselV = LDBL_MAX;
      }
      p1n1 = std::pow(x, 2*n - m);
      p1n2 = std::pow(absX, m - n);
      if(p1n2 == 0){p1n2 = std::numeric_limits< long double >::denorm_min();}
      if(std::isinf(p1n2)){p1n2 = LDBL_MAX;}
      p1n3 = std::pow(std::sqrt(sig1), m - n - 1);
      p1d1 = std::pow(1 - std::pow(p, 2), 2*n + 0.5);
      p1d2 = std::pow(std::sqrt(sig2), m - n +1);
      p3 = std::pow(mu1/sig1 - p*mu2/sig1sig2, m);
      p4 = std::pow(mu2/sig2 - p*mu1/sig1sig2, 2*n - m);

      sumT += (p1n1*p1n2*p1n3) / (M_PI * factN2 * p1d1*p1d2) * Rf_choose(2*n, m) * p3 * p4 * besselV;

    }
  }
  if(sumT == R_NaN){sumT = maxRet;}
  long double toReturn = sumT * exp((-1 / (2 * (1 - std::pow(p, 2))))*
                                    (std::pow(mu1, 2)/sig1 + std::pow(mu2, 2)/sig2 - (2*p*(x + mu1*mu2))/sig1sig2));
  if(toReturn > maxRet){return maxRet;}
  return toReturn;
}

class exProdPDF : public Func
{
private:
  long double mu1_in;
  long double mu2_in;
  long double mu3_in;
  long double sig1_in;
  long double sig2_in;
  long double sig3_in;
  long double sig12_in;
  long double sig13_in;
  long double sig23_in;
  double indL;
  long double maxRet;
public:
  exProdPDF(long double mu1_in_, long double mu2_in_, long double mu3_in_,
            long double sig1_in_, long double sig2_in_, long double sig3_in_,
            long double sig12_in_, long double sig13_in_, long double sig23_in_,
            double indL_,
            long double maxRet_) : mu1_in(mu1_in_), mu2_in(mu2_in_), mu3_in(mu3_in_),
            sig1_in(sig1_in_), sig2_in(sig2_in_), sig3_in(sig3_in_),
            sig12_in(sig12_in_), sig13_in(sig13_in_), sig23_in(sig23_in_),
            indL(indL_), maxRet(maxRet_){}

  double operator()(const double& x) const
  {
    //// Now set up for interactions
    // mu1 is for alpha_G, mu2 for gamma, mu3 for psi.
    // If psi is zero, then no interaction and same as before

    /// Set values to mu1 and mu2, sig1, sig2, sig12 as used previously
    long double mu1 = mu1_in;
    long double sig1 = sig1_in;

    long double mu2 = mu2_in + indL*mu3_in;
    long double sig2 = sig2_in + std::pow(indL, 2.0)*sig3_in +
      2*indL*sig23_in;
    long double sig12 = sig12_in + indL*sig13_in;


    /// All after here is the same as before
    long double sig1sig2 = std::sqrt(sig1*sig2);
    long double p = sig12 / sig1sig2;

    if(x == 0){
      long double toReturn = ((long double)std::pow(std::sqrt(sig1), - 1)) /
        (long double)(M_PI * (long double)std::pow(1 - std::pow(p, 2), 0.5)*(long double)std::sqrt(sig2)) * LDBL_MAX;
      if(toReturn > maxRet){return maxRet;}
      return toReturn;
    }

    long double absX = fabs(x);

    long double sumT = 0;
    //double factN = 1;
    double factN2 = 1;
    long double besselV = 0;
    long double p1n1, p1n2, p1n3, p1d1, p1d2, p3, p4;
    for(int n = 0; n <= 30; n++){
      if(n > 0){
        //factN *= n;
        factN2 *= 2*n * (2*n-1);
      }
      for(int m = 0; m <= 2*n; m++){
        try{
          besselV = boost::math::cyl_bessel_k((long double)(m - n), (long double)(std::abs(x)/((1 - std::pow(p, 2))*sig1sig2)));
        }
        catch (std::exception& ex){
          besselV = LDBL_MAX;
        }
        p1n1 = std::pow(x, 2*n - m);
        p1n2 = std::pow(absX, m - n);
        if(p1n2 == 0){p1n2 = std::numeric_limits< long double >::denorm_min();}
        if(std::isinf(p1n2)){p1n2 = LDBL_MAX;}
        p1n3 = std::pow(std::sqrt(sig1), m - n - 1);
        p1d1 = std::pow(1 - std::pow(p, 2), 2*n + 0.5);
        p1d2 = std::pow(std::sqrt(sig2), m - n +1);
        p3 = std::pow(mu1/sig1 - p*mu2/sig1sig2, m);
        p4 = std::pow(mu2/sig2 - p*mu1/sig1sig2, 2*n - m);

        sumT += (p1n1*p1n2*p1n3) / (M_PI * factN2 * p1d1*p1d2) * Rf_choose(2*n, m) * p3 * p4 * besselV;

      }
    }
    if(sumT == R_NaN){sumT = maxRet;}
    long double toReturn = sumT * exp((-1 / (2 * (1 - std::pow(p, 2))))*
                                      (std::pow(mu1, 2)/sig1 + std::pow(mu2, 2)/sig2 - (2*p*(x + mu1*mu2))/sig1sig2));
    if(toReturn > maxRet){return maxRet;}
    return toReturn;
  }
};

// [[Rcpp::export]]
double exFindLower(long double mu1, long double mu2, long double mu3,
                   long double sig1, long double sig2, long double sig3,
                   long double sig12, long double sig13, long double sig23, double indL = 1, double pdfTol = 1e-5){
  double shift = fabs((std::pow(mu1, 2)*sig1 + std::pow(mu2, 2)*sig2 + std::pow(mu3, 2)*sig3 + sig1*sig2*sig3) +
                      std::abs(mu1*(mu2+indL*mu3)));
  if(shift == 0){
    shift += 0.001;
  }
  double x = mu1*(mu2 + indL*mu3);
  while(prodExPDF(x, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL) > pdfTol &
        x > x - 1e5*shift){
    x -= shift;
  }

  //Rcout << "Lower: " << x << "\n\n";
  return x;
}
// [[Rcpp::export]]
double exFindUpper(long double mu1, long double mu2, long double mu3,
                   long double sig1, long double sig2, long double sig3,
                   long double sig12, long double sig13, long double sig23, double indL = 1, double pdfTol = 1e-5){

  double shift = fabs((std::pow(mu1, 2)*sig1 + std::pow(mu2, 2)*sig2 + std::pow(mu3, 2)*sig3 + sig1*sig2*sig3) +
                      std::abs(mu1*(mu2+indL*mu3)));
  if(shift == 0){
    shift += 0.001;
  }
  double x = mu1*(mu2 + indL*mu3);
  //Rcout << "Upper Step: " << x << "\n";
  while(prodExPDF(x, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL) > pdfTol &
        x < x + 1e5*shift){
    x += shift;
  }

  //Rcout << "Upper: " << x << "\n\n";
  return x;
}


// [[Rcpp::export]]
long double exProdCDF(long double x,
                      long double mu1, long double mu2, long double mu3,
                      long double sig1, long double sig2, long double sig3,
                      long double sig12, long double sig13, long double sig23,
                      long double maxRet = 1e300, double indL = 1){
  exProdPDF f(mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL, maxRet);
  double err_est;
  int err_code;
  const double upper = exFindUpper(mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  if(upper < x){return 1.0;}
  const double lower = exFindLower(mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  const long double res = integrate(f, lower, (long double)x, err_est, err_code);
  return res;
}


// [[Rcpp::export]]
arma::vec exProdCI(long double alpha,
                   long double mu1, long double mu2, long double mu3,
                   long double sig1, long double sig2, long double sig3,
                   long double sig12, long double sig13, long double sig23,
                   long double maxRet = 1e300, double indL = 1, long double citol = 1e-8){
  arma::vec CI = arma::zeros(2);
  double deltaEst = mu1 * (mu2 * indL*mu3);
  double curGuess = deltaEst;
  long double curStep = std::max(fabs(mu1), std::max(fabs(mu2), fabs(mu2 + mu3)));
  double curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
  //Rcout << "Bad Loop\n";
  while(curCP < alpha/2.0){
    curGuess += curStep;
    curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
    //Rcout << "Guess: " << curGuess << ", CDF: " << curCP << "\n";
  }
  //Rcout << "Start Lower Loop\n";
  while(fabs(curCP - alpha/2.0) > citol){
    while(curCP > alpha/2.0){
      curGuess -= curStep;
      curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
      //Rcout << "Guess: " << curGuess << ", CDF: " << curCP << "\n";
    }
    curGuess += curStep;
    curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
    curStep /= 2;
  }
  CI(0) = curGuess;

  curGuess = deltaEst;
  curStep = std::max(fabs(mu1), std::max(fabs(mu2), fabs(mu2 + mu3)));
  curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
  while(curCP > (1-alpha)/2.0){
    curGuess -= curStep;
    curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
  }
  //Rcout << "Start Uppere Loop\n";
  while(fabs(curCP - (1 - alpha)/2.0) > citol){
    while(curCP < (1-alpha)/2.0){
      curGuess += curStep;
      curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
      //Rcout << "Guess: " << curGuess << ", CDF: " << curCP << "\n";
    }
    curGuess -= curStep;
    curCP = exProdCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, maxRet, indL);
    curStep /= 2;
  }
  CI(1) = curGuess;
  return(CI);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/////////////////////Product of Normal RV - Not Exact///////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// For these, main integration happens across x (first variable alpha_G)
/// We change the value of z though, which takes the place of y in a way

// [[Rcpp::export]]
double mySign(double x)
{
  if(x >= 0){return 1;}else{return -1;}
}

// [[Rcpp::export]]
double prodPDF_c(double x, long double z,
                 long double mu1, long double mu2, long double mu3,
                 long double sig1, long double sig2, long double sig3,
                 long double sig12, long double sig13, long double sig23, double indL = 1){
  long double muU = mu1;
  long double sigU = sig1;
  long double muX = muU / std::sqrt(sigU);

  long double muV = mu2 + indL*mu3;
  long double sigV = sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23;
  long double muY = muV / std::sqrt(sigV);

  long double sigUV = sig12 + indL*sig13;

  long double sigUsigV = std::sqrt(sigU*sigV);
  long double p = sigUV / sigUsigV;

  long double muYcX = muY + p*(x-muX);

  long double p1 = R::dnorm(x - muX, 0.0, 1.0, 0);
  long double p2 = R::pnorm(mySign(x) * (( (z/std::max(x, .0000000001)) - muYcX ) / std::sqrt(1-std::pow(p, 2.0))), 0.0, 1.0, 1, 0);
  return( p1 * p2 );
}

class prodPDF : public Func
{
private:
  long double z;
  long double mu1;
  long double mu2;
  long double mu3;
  long double sig1;
  long double sig2;
  long double sig3;
  long double sig12;
  long double sig13;
  long double sig23;
  double indL;
public:
  prodPDF(long double z_, long double mu1_, long double mu2_, long double mu3_,
          long double sig1_, long double sig2_, long double sig3_,
          long double sig12_, long double sig13_, long double sig23_,
          double indL_) : z(z_), mu1(mu1_), mu2(mu2_), mu3(mu3_),
          sig1(sig1_), sig2(sig2_), sig3(sig3_),
          sig12(sig12_), sig13(sig13_), sig23(sig23_),
          indL(indL_){}

  double operator()(const double& x) const
  {long double muU = mu1;
    long double sigU = sig1;
    long double muX = muU / std::sqrt(sigU);

    long double muV = mu2 + indL*mu3;
    long double sigV = sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23;
    long double muY = muV / std::sqrt(sigV);

    long double sigUV = sig12 + indL*sig13;

    long double sigUsigV = std::sqrt(sigU*sigV);
    long double p = sigUV / sigUsigV;

    long double muYcX = muY + p*(x-muX);
    long double p1 = R::dnorm(x - muX, 0.0, 1.0, 0);
    long double p2 = R::pnorm(mySign(x) * ( (z/std::max(x, .0000000001) - muYcX ) / std::sqrt(1-std::pow(p, 2.0))), 0.0, 1.0, 1, 0);
    return( p1 * p2 );
  }
};

// [[Rcpp::export]]
double findLower(long double z,
                 long double mu1, long double mu2, long double mu3,
                 long double sig1, long double sig2, long double sig3,
                 long double sig12, long double sig13, long double sig23, double indL = 1){

  double shift = fabs(mu1 / sig1 / 1000.0);
  if(shift == 0){
    shift += 0.0000001;
  }

  long double muV = mu2 + indL*mu3;
  long double sigV = sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23;
  long double muY = muV / std::sqrt(sigV);

  //Rcout << "muV: " << muV << ", sigV: " << sigV << ", muY: " << muY << "\n" <<
  //          "search start: " <<  muY - 3*std::sqrt(sigV) << "\n";

  double x = muY - 3*std::sqrt(sigV);
  while(prodPDF_c(x, z, mu1, mu2, mu3,
                  sig1, sig2, sig3,
                  sig12, sig13, sig23, indL) > 1e-14 &
                    x > x - 1e5*shift){
    //Rcout << "Lower Search\n";
    x -= shift;
  }

  //Rcout << "lower: " << x << "\n";
  return x;
}
// [[Rcpp::export]]
double findUpper(long double z,
                 long double mu1, long double mu2, long double mu3,
                 long double sig1, long double sig2, long double sig3,
                 long double sig12, long double sig13, long double sig23, double indL = 1){
  double shift = fabs(mu1 / sig1 / 1000.0);
  if(shift == 0){
    shift += 0.0000001;
  }

  long double muV = mu2 + indL*mu3;
  long double sigV = sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23;
  long double muY = muV / std::sqrt(sigV);

  //Rcout << "muV: " << muV << ", sigV: " << sigV << ", muY: " << muY << "\n" <<
  //  "search start: " <<  muY + 3*std::sqrt(sigV) << "\n";

  double x = muY + 3*std::sqrt(sigV);
  while(prodPDF_c(x, z, mu1, mu2, mu3,
                  sig1, sig2, sig3,
                  sig12, sig13, sig23, indL) > 1e-14 &
                    x < x + 1e5*shift){
    //Rcout << "Upper Search\n";
    x += shift;
  }
  //Rcout << "upper: " << x << "\n";
  return x;
}


// [[Rcpp::export]]
double prodCDF(long double z,
               long double mu1, long double mu2, long double mu3,
               long double sig1, long double sig2, long double sig3,
               long double sig12, long double sig13, long double sig23,
               double indL = 1){
  prodPDF f(z, mu1, mu2, mu3,
            sig1, sig2, sig3,
            sig12, sig13, sig23, indL);
  const double lower = findLower(z, mu1, mu2, mu3,
                                 sig1, sig2, sig3,
                                 sig12, sig13, sig23, indL);
  const double upper = findUpper(z, mu1, mu2, mu3,
                                 sig1, sig2, sig3,
                                 sig12, sig13, sig23, indL);
  double err_est;
  int err_code;

  const double res = integrate(f, lower, upper, err_est, err_code);
  return res;
}


// [[Rcpp::export]]
arma::vec prodCI(long double alpha,
                 long double mu1, long double mu2, long double mu3,
                 long double sig1, long double sig2, long double sig3,
                 long double sig12, long double sig13, long double sig23,
                 long double maxRet = 1e300, double indL = 1, long double citol = 1e-8){

  arma::vec CI = arma::zeros(2);
  double deltaEst = mu1 * (mu2 * indL*mu3);
  double curGuess = deltaEst;
  long double curStep = std::max(fabs(mu1), std::max(fabs(mu2), fabs(mu2 + mu3)));
  double curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  while( curCP < alpha/2.0){
    curGuess += curStep;
    curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  }
  while(fabs(curCP - alpha/2.0) > citol){
    while(curCP > alpha/2.0){
      curGuess -= curStep;
      curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
      Rcout << "Guess: " << curGuess << ", CP: " << curCP << "\n";
      ::sleep(2);
      //Rcout << "Lower Loop It\n";
    }
    curGuess += curStep;
    curStep /= 2;
    curCP = .5;
  }
  CI(0) = curGuess * std::sqrt(sig1)*std::sqrt(sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23);
  curGuess = deltaEst;
  curStep = std::max(fabs(mu1), std::max(fabs(mu2), fabs(mu2 + mu3)));
  curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  while(curCP > (1-alpha)/2.0){
    curGuess -= curStep;
    curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
  }
  while(fabs(curCP - (1 - alpha)/2.0) > citol){
    while(curCP < (1-alpha)/2.0){
      curGuess += curStep;
      curCP = prodCDF(curGuess, mu1, mu2, mu3, sig1, sig2, sig3, sig12, sig13, sig23, indL);
      //Rcout << "Upper Loop It\n";
    }
    curGuess -= curStep;
    curStep /= 2;
    curCP = .5;
  }
  CI(1) = curGuess * std::sqrt(sig1)*std::sqrt(sig2 + std::pow(indL, 2.0)*sig3 + 2*indL*sig23);
  return(CI);
}
