import numpy as np
import scipy.constants as cns
import scipy.integrate as intg

# Wavelengths of two prominent broad lines

Hbeta = 4.86135e-7
Halpha = 6.56281e-7

#%%
class Gauss:
    def __init__(self, line):
        self.line = line
        if line == 'Halpha':
            self.x1 = Halpha
        elif line == 'Hbeta':
            self.x1 = Hbeta        
       
    def fgmod(self, x, x1, sigma, norm):
        self.x = x
        self.sigma = sigma
        self.x1 = x1
        self.norm = norm
        exponent_gauss = 0.5*((self.x - self.x1)/self.sigma)**2
        ff = self.norm*np.exp(-exponent_gauss)
        return ff
    
#%%  
class diskline:
    def __init__(self, line):
        self.line= line
        if self.line == 'Halpha':
            self.x0= Halpha
        elif self.line == 'Hbeta':
            self.x0= Hbeta
        elif self.line == 'other':
            self.x0= float(input("Specify wavelength in m"))
            
        self.normalization= cns.G
        self.dphi= 0.005
        self.xnum= 250
        self.emissivity_index = 3
        
    def epsilon1(self, xi):
        """Emissivity function - radial"""
        ff= xi**-self.emissivity_index 
        return ff
    
    def D(self, ri, phi, i):
        ff= np.sqrt( (1 - 3/ri) ) / ( 1 + np.sin(i)*np.sin(phi)/ri**0.5 )
        return ff

    def G(self, ri, phi, i):
        D_eval = self.D(ri, phi, i) # Storing the function evaluation here 
        Denominator = D_eval**2 * np.cos(i)**2 + ri * (D_eval - np.sqrt(1 - 3/ri))**2
        ff = 1 + (1/ri) * ( 2*D_eval**2 / Denominator  -  1)
        return ff

    def integrand(self, nu, lam0, factor,
                  ri, phi, theta_i):
        """Combines the emissivity, local broadening, Doppler effect, light-bending in this function"""
        i = theta_i* np.pi/180 # angle of inclination (converted to inclination) used first for calculation from here, 
        nu0 = cns.c/lam0
        D_eval = self.D(ri, phi, i)
        sigma0 = factor*nu0
        X = (nu/nu0) - 1
        ff = ri*self.epsilon1(ri)* np.exp(-(1 + X - D_eval)**2*nu0**2/(2*sigma0**2*D_eval**2))* D_eval**3* self.G(ri, phi, i)
        return ff
    
    def calc_xarray(self, r_in, r_out):
        """
        Function calculates the array with between inner radius and outer radius.
        The intervals are different for different radius intervals.
        The integrand is evaluated more densely for xin < 50Rg        
        """
        if r_in >= 50:
            xarr = np.linspace(r_in, r_out, self.xnum)
            return xarr
        elif r_in < 50:
            x2 = np.linspace(50, r_out, self.xnum)
            if r_in < 10:
                xup = np.array( [10,20,30,40,50] )
                x11 = np.arange( r_in,   xup[0], 0.08 )
                x12 = np.arange( xup[0], xup[1], 0.1)
                x13 = np.arange( xup[1], xup[2], 0.2)
                x14 = np.arange( xup[2], xup[3], 0.3)
                x15 = np.arange( xup[3], xup[4], 0.4)
                x1 = np.concatenate((x11, x12, x13, x14, x15))
            if r_in >= 10 and r_in < 20:
                xup = np.array([20,30,40,50])
                x11 = np.arange( r_in,   xup[0], 0.1)
                x12 = np.arange( xup[0], xup[1], 0.2)
                x13 = np.arange( xup[1], xup[2], 0.3)
                x14 = np.arange( xup[2], xup[3], 0.4)
                x1 = np.concatenate((x11, x12, x13, x14))
            if r_in >= 20 and r_in < 30:
                xup = np.array([30,40,50])
                x11 = np.arange( r_in, xup[0], 0.2)
                x12 = np.arange( xup[0], xup[1], 0.3)
                x13 = np.arange( xup[1], xup[2], 0.4)
                x1 = np.concatenate((x11, x12, x13))
            if r_in >= 30 and r_in < 40:
                xup = np.array([40,50])
                x11 = np.arange(r_in, xup[0], 0.3)
                x12 = np.arange(xup[0], xup[1], 0.4)
                x1 = np.concatenate((x11, x12))
            if r_in >= 40:
                x1 = np.arange(r_in,50,0.5)
            xarr = np.concatenate((x1, x2))
            return xarr
    
  
    def Fnu(self, nu, lam0, factor,
            r_in, r_out, theta_i):
        """ Evaluate flux density at each frequency """
        phi  = np.arange(-np.pi/2, np.pi/2, self.dphi) # array for angular integration
        ri = self.calc_xarray(r_in, r_out) # array for radial integration
        ii_xi = []
        
        for i in range(ri.size):
            ii_phi = self.integrand( nu,   lam0, factor, 
                                     ri[i], phi, theta_i )
            ii0 = intg.simps( ii_phi, dx=self.dphi) #integrate for phi 
            ii_xi.append(ii0)
            
        ii_xi = np.array(ii_xi)
        ff = intg.simps( ii_xi, x=ri ) #integrate for radius
        return ff


    def F(self, nu, lam0, factor,
          r_in, r_out, theta_i):
        """Evaluate intensity at each frequency """
        ff = np.zeros(nu.size, dtype = np.float)
        for i in range(nu.size):
            ff[i] = self.Fnu( nu[i], lam0, factor, 
                              r_in,  r_out, theta_i)
        return ff

    
    def flam(self, lamda, lam0, factor,
             r_in, r_out, theta_i, norm):
        """ The frequency array transformed wavelength array. The scale factor is introduced to make the normalization ~ 0.1.
        This is required for fitting purposes.
        For data fitting this value should be equated with the expression of Chen and Halpern.
        """
        nu = cns.c/lamda
        ff = self.F( nu,    lam0, factor,
                     r_in, r_out, theta_i)/ lamda**2
        scale_factor = 1e-9
        return scale_factor*norm*ff
