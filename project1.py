
import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from prettytable import PrettyTable


if len (sys.argv ) != 2 :
    print("usage is : %s input[text file] " % (sys.argv[0])) 
    sys.exit()

input_file= open(sys.argv[1], 'r')
for i, line in enumerate(input_file):
    if i == 0:
        comment = line.strip()
        print("comment = {}\n".format(comment))
    elif i == 1:
        eqn_of_state = line.strip()
        print("equation of state to be used =  {}".format(eqn_of_state))
    elif i== 2:
        parameters = np.array([float(x) for x in line.split()])
        print("number of parameters= {}".format(len(line.strip().split())))
        print("initial guess = {}".format(parameters))
    elif i == 3:
        line1 =line.strip('\n')
        temp = float(line1.split()[1])
        print(line1)
    elif i >= 4:
        molar_volume_unit = line.strip().split()[0];
        print("Molar Volume Unit= {}".format(molar_volume_unit))
        pressure_unit =line.strip().split()[1]
        print("pressure unit= {}".format(pressure_unit))       
        data = [d.split() for d in input_file]
print("number of data points =", len(data)) 
print("\nthe data=") 
values = [[float(x) for x  in sublist] for sublist in data]
data_array=np.array(values)
print(data_array[:10,:])
Data_SI = np.zeros(data_array.shape)

#converting to SI units:
#For volume:

if  molar_volume_unit.lower() == "l/mol" or molar_volume_unit.lower() == "dm^3/mol":
    Data_SI[:,0] = (1e-3)*data_array[:,0]
elif molar_volume_unit.lower() == "cm^3/mol":
    Data_SI[:,0] = (1e-6)*data_array[:,0]
elif molar_volume_unit.lower() == "m^3/mol":
    Data_SI[:,0] = data_array[:,0]
else:
    print("Please Enter a valid file with a valid units of volume")

 #for pressure    
if  pressure_unit.lower() == "torr" :
    Data_SI[:,1] = (101325/760)*data_array[:,1]
elif pressure_unit.lower() == "atm":
    Data_SI[:,1] = (101325)*data_array[:,1]
elif pressure_unit.lower() == "mmhg":
    Data_SI[:,1] = (133.322387415)*data_array[:,1]
elif pressure_unit.lower() == "bar":
    Data_SI[:,1] = (10e5)*data_array[:,1]
elif pressure_unit.lower() == "kilobar" or pressure_unit.lower() == "kbar":
    Data_SI[:,1] = (10e8)*data_array[:,1]
elif pressure_unit.lower() == "pa":
    Data_SI[:,1] = data_array[:,1]
else:
    print("Please Enter a valid file with a valid units of pressure")

print("\ndata converted to SI units = \n ", Data_SI[:10,:])
R =8.31446261815324

def jacobian(a_):
    if eqn_of_state == 'vdW':
        first_column = -1/(Data_SI[:,0]**2)
        second_clmn=(R*temp)/(Data_SI[:,0] - a_[1])**2 
        jacobian_matrix = np.stack((first_column, second_clmn), axis=-1)
        return jacobian_matrix

    elif eqn_of_state == 'berthelot':
        first_column = -1/(temp*(Data_SI[:,0]**2))
        second_clmn=(R*temp)/((Data_SI[:,0] - a_[1])**2) 
        jacobian_matrix = np.stack((first_column, second_clmn), axis=-1)
        return jacobian_matrix  

    elif eqn_of_state == 'Dieterici': 
        power = -a_[0]/(R*temp*Data_SI[:,0])
        first_column = -(np.exp(power)/Data_SI[:,0])/(Data_SI[:,0] - a_[1])
        second_clmn= (R*temp)*(np.exp(power))/(Data_SI[:,0] - a_[1])**2
        jacobian_matrix = np.stack((first_column, second_clmn), axis=-1)
        return jacobian_matrix

    elif eqn_of_state == 'rk':
        first_column = -1/(math.sqrt(temp)*Data_SI[:,0]*(Data_SI[:,0] + a_[1]))
        second_clmn=(R*temp)/(Data_SI[:,0] - a_[1])**2 + \
        a_[0]/(math.sqrt(temp)*Data_SI[:,0]*(Data_SI[:,0] - a_[1])) 
        jacobian_matrix = np.stack((first_column, second_clmn), axis=-1)
        return jacobian_matrix

def f_calculated(a_):
    if eqn_of_state == 'vdW':
        f = ((R*temp)/(Data_SI[:,0] - a_[1]) - \
                                (a_[0]/(Data_SI[:,0])**2))
        return f

    elif eqn_of_state == 'berthelot':
        f = ((R*temp)/(Data_SI[:,0] - a_[1]) - \
                             (a_[0]/((Data_SI[:,0]**2)*temp)))
        return f

    elif eqn_of_state == 'Dieterici': 
        power = -a_[0]/(R*temp*Data_SI[:,0])
        f = (R*temp*np.exp(power))/(Data_SI[:,0] - a_[1])
        return f
   
    elif eqn_of_state == 'rk':
        f = ((R*temp)/(Data_SI[:,0] - a_[1]) \
                            - (a_[0]/(math.sqrt(temp)*Data_SI[:,0]*(Data_SI[:,0] +a_[1]))))
        return f

def delta_y(a_): 
    
    delta_y_ = Data_SI[:,1] - f_calculated(a_)
    return delta_y_

    
def S(a_):
    error_s = np.dot(np.transpose(delta_y(a_)),delta_y(a_))
    return error_s

initial_error = S(parameters)
print("\n\n\n Inintial Error =",initial_error)
print("##############################################") 
Lambda = math.pow(10,4)
cycle_number =0   #intializing cirlce number
change = initial_error #initializing error with the initial value
a_ = parameters  #matrix a cosisting of parameters
Lambda = 1.00e4        #initial Lambda value  

while(change>1e-4):
    
    cycle_number = cycle_number +1
    print("Cycle Number=",cycle_number)
    print("Lambda",Lambda)
    alpha = np.dot(np.transpose(jacobian(a_)),jacobian(a_))
    beta_function = np.dot(np.transpose(jacobian(a_)),delta_y(a_))

    alpha_prime_function = np.zeros([alpha.shape[0],alpha.shape[1]])   #declaring an array of zeros
    
    for i in range(alpha.shape[0]):
        for j in range(alpha.shape[1]):
            if i == j:
                alpha_prime_function[i, j] = alpha[i,j] * (Lambda + 1)
            else:
                alpha_prime_function[i,j] = alpha[i,j]
                
    old_S = S(a_)
    delta_a_transposed = np.dot(np.linalg.inv(alpha_prime_function),beta_function)
    delta_a = np.transpose(delta_a_transposed)
    a_= a_ + delta_a
    new_S= S(a_)
    change =abs(old_S-new_S)
    if new_S > old_S:
        Lambda = Lambda*10
    else:
        Lambda = Lambda/10
        
    print("Beta\n",beta_function)
    print("alpha\n", alpha)
    print("alph'\n",alpha_prime_function)
    print('new a', a_)
    print('change_a',np.transpose(delta_a))
    print('old_S',old_S)
    print('new_S', new_S)
    print("change",change)
    print("------------------------------")
    
fitted_values = f_calculated(a_)    
print("===========final statistics========")
print("chi**2 = {}".format(new_S))
sigma_square = new_S/(Data_SI.shape[0]-a_.shape[0])
print("sigma**2 = {}".format(sigma_square))
print("alpha = \n {}".format(alpha_prime_function))
c = np.linalg.inv(alpha_prime_function)
print("C = \n {}".format(c))
print("comment:",comment)
#rho = 
x = PrettyTable()

x.field_names = ["No.", "parameter", "Standard Error"]
rho1= math.sqrt((new_S**2)*c[0,0])
rho2= math.sqrt((new_S**2)*c[1,1])
x.add_row([1, a_[0],rho1])
x.add_row([2, a_[1],rho2])
print(x)
rho = math.sqrt((new_S**2)*c[0,1])/(rho1*rho2)
print('rho(1,2)=',rho)
P_mean = np.sum(Data_SI, axis=0)[1]/Data_SI.shape[0]
ss_tot = np.sum((Data_SI-P_mean)**2,axis =0)[1]
print("P_mean= {}, SS_tot = {}".format(P_mean,ss_tot))
determination_coefficient = 1- (new_S/ss_tot)
print("Coefficient of determination     =", determination_coefficient)
adjusted_corrl_coeff = 1- ((new_S/(Data_SI.shape[0]-a_.shape[0]))/(ss_tot/(a_.shape[0])))
print("adjusted correlation coefficient =", adjusted_corrl_coeff)
tot1 =[]
tot2 = []
for i in range(Data_SI.shape[0]):
    tot1.append(abs(S(a_)))
    tot2.append(abs(Data_SI[:,1]))
R_factor = sum(tot1)/np.sum(tot2)    
print("R factor                         =",R_factor,"%")
plt.plot(Data_SI[:,0], Data_SI[:,1], 'r', marker = 'o', label='observed');
plt.plot(Data_SI[:,0],fitted_values, 'g', label = 'fitted')
plt.style.use('seaborn')
plt.title('Vm versus pressure')
plt.xlabel('Vm')
plt.ylabel('pressure')
plt.legend()
plt.show()