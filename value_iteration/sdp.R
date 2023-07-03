
# state space limit
Nmax <- 200
states <- seq(0, 2, length.out = Nmax)
# Vector of actions: rate of the population that can be removed, ranging #from 0 to 1
actions <- states # seq(0, 1, 1/(Nmax+1))

# Population growth rate
r <- 0.25
K <- 1

# Function for the exponential growth of the dynamic model
dynamic <- function(x, a) {
  y <- pmax(x - a, 0)  #(1-a)*x
  x_t1 <- y + y * r * (1 - y / K)
  if(x_t1 < 0) x_t1 <- 0
  x_t1
}

# Utility function
get_utility <- function(x, a) {
  # a * x
  pmin(x,a)
}


prob <- function(states, mu, sigma){
  n_s <- length(states)
  meanlog <- log(mu) - sigma ^ 2 / 2
  x <- dlnorm(states, meanlog, sigma)
  N <- plnorm(states[n_s], meanlog, sigma)

  ## Handle exceptions (e.g. from mu ~ 0)
  if(sum(x) == 0){  ## nextpop is computationally zero, would create NAs
    x <- c(1, rep(0, n_s - 1))
    ## Normalize, pile on boundary
  } else {
    x <- x * N / sum(x)         # normalize densities to  = cdf(boundary)
    x[n_s] <- 1 - N + x[n_s]    # pile remaining probability on boundary
  }
  x
}


transition <- array(0, dim = c(length(states), length(states), length(actions)))
utility <- array(0, dim = c(length(states), length(actions)))
sigma = 0.01
for (k in 1:Nmax) {
  # Loop on all actions
  for (i in 1:length(actions)) {
    nextpop <- dynamic(states[k], actions[i])
    if(nextpop <= 0){
      transition[k, , i] <- c(1, rep(0, Nmax-1))
    } else {
      transition[k, , i] <- prob(states, nextpop, sigma)
    }
    utility[k,i] <- get_utility(nextpop, actions[i])
  } # end of action loop
} # end of state loop



# Discount factor
discount <- 0.98

library(MDPtoolbox)

soln <- MDPtoolbox::mdp_value_iteration(transition, utility, discount)
#soln <- MDPtoolbox::mdp_policy_iteration(transition, utility, discount)
escapement = states - states * actions[soln$policy]
plot(states, actions[soln$policy])
plot(states[1:100], escapement[1:100])








# Action value vector at tmax
Vtmax <- numeric(length(states))

# Action value vector at t and t+1
Vt <- numeric(length(states))
Vtplus <- numeric(length(states))

# Optimal policy vector
D <- numeric(length(states))

# Time horizon
Tmax <- 200

# The backward iteration consists in storing action values in the vector Vt which is the maximum of
# utility plus the future action values for all possible next states. Nmaxnowing the final action
# values, we can then backwardly reset the next action value Vtplus to the new value Vt. We start
# The backward iteration at time T-1 since we already defined the action #value at Tmax.
for (t in (Tmax-1):1) {

  # We define a matrix Q that stores the updated action values for #all states (rows)
  # actions (columns)
  Q <- array(0, dim=c(length(states), length(actions)))

  for (i in 1:length(actions)) {

    # For each harvest rate we fill for all states values (row)
    # the ith column (Action) of matrix Q
    # The utility of the ith action recorded for all states is
    # added to the product of the transition matrix of the ith
    # action by the action value of all states
    Q[,i] <- utility[,i] + discount*(transition[,,i] %*% Vtplus)

  } # end of the harvest loop

  # Find the optimal action value at time t is the maximum of Q
  Vt <- apply(Q, 1, max)

  # After filling vector Vt of the action values at all states, we
  # update the vector Vt+1 to Vt and we go to the next step standing
  # for previous time t-1, since we iterate backward
  Vtplus <- Vt

} # end of the time loop

# Find optimal action for each state
policy <- numeric(length(states))

for (k in 1:Nmax) {
  # We look for each state which column of Q corresponds to the
  # maximum of the last updated value
  # of Vt (the one at time t+1). If the index vector is longer than 1
  # (if there is more than one optimal value we chose the minimum action)
  policy[k] <- min(which(Q[k,] == Vt[k]))
  D[k] <- actions[policy[k]]
}

##################################################################################
# PLOT SOLUTION
##################################################################################

plot(states, D, xlab="Population size", ylab="harvest rate")

escapement <- states - D
plot(states[1:100], escapement[1:100], xlab="Population size", ylab="escapement")
