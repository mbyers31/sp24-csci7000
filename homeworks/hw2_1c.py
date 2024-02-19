import numpy as np


def error(i_euler, i_analytical):
    '''
    Computes the error between the numerical and analytical solutions.
    '''
    difference = np.subtract(i_euler, i_analytical)
    return np.max(np.abs(difference))


'''
Below are the two functions used for solving i analytically.
'''
def i_analytical(i_init, beta, gamma):
    '''
    Analytical solution for i
    '''
    #we have a closed form solution for i
    R = beta / gamma
    numerator = 1 - (1 / (R))
    factor = (1 - (1 / R) - i_init) / i_init
    return lambda t: numerator / (1 + factor * np.exp(-(beta-gamma)*t))


def solve_analytical(init_conditions, delta_t, max_t, beta, gamma):
    '''
    Solves the analytical solution
    '''
    t = [init_conditions[0]]
    #we don't care about s_0 here
    i = [init_conditions[2]]

    idx = 0

    #set up i's analytical solution
    i_solution = i_analytical(i[0], beta, gamma)

    while t[idx] < max_t:
        t.append(t[idx] + delta_t)
        i.append(i_solution(t[idx+1]))
        idx += 1

    return t, i


'''
Below are the functions used for solving i numerically
'''
def update(t, delta_t, current_val, f_prime):
    return current_val + (delta_t * f_prime(current_val))


#define derivatives
def s_dot(beta, I, gamma):
    return lambda S : (-1 * beta * S * I) + (gamma * I)


def i_dot(beta, gamma, S):
    return lambda I : (beta * S * I) - (gamma * I)


#solver
def solve(init_conditions, delta_t, max_time, beta, gamma):
    #set up experiment loop
    t = [init_conditions[0]]
    S =[init_conditions[1]]
    I = [init_conditions[2]]
    
    idx=0
    
    while t[idx] < max_time:
        #create derivatives
        s_update = s_dot(beta, I[idx], gamma)
        i_update = i_dot(beta, gamma, S[idx])
        
        #update S, I, R
        S.append(update(t[idx], delta_t, S[idx], s_update))
        I.append(update(t[idx], delta_t, I[idx], i_update))
    
        #upate time
        t.append(t[idx] + delta_t)
        idx += 1

    return t, S, I


'''
The solver to find both solutions and compute the error.
'''
def solve_analytical_vs_numerical():
    #set up initial conditions
    s_init = 0.99
    i_init = 0.01
    t_init = 0

    beta = 3
    gamma = 2

    delta_ts = [2, 1, 1/2, 1/4, 1/8, 1/16, 1/32]
    max_t = 25

    init_conditions = [t_init, s_init, i_init]
    errors = []

    #solve loop
    for delta_t in delta_ts:
        t, _, i_euler = solve(init_conditions, delta_t, 
                              max_t, beta, gamma)
        t, i_analytical_sol = solve_analytical(init_conditions, delta_t, 
                                               max_t, beta, gamma)

        errors.append(error(i_analytical_sol, i_euler))
    
    return errors


def main():
    errors = solve_analytical_vs_numerical()
    print(errors)
        

if __name__ == "__main__":
    main()