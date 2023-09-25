import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import ascii
from astropy.table import Table

def gal_orb(rh, lon, lat, vr, pml, pmb, t0, tf, M_disc = 100.0, M_sph = 30.0, rtol=1e-12, atol=1e-12, name = None, reverse = False, plot = False,
            output = None):
    """
    Calculating the orbit of star or stellar cluster in the gravitational field of the Galaxy described 
    by three-component galaxy: disc, spheroid and halo.

    Parameters
    ----------
    rh : float
        Heliocentric distance of the object (in kpc).
    lon : float
        Galactic longitude of the object (in degrees).
    lat : float
        Galactic latitude of the object (in degrees).
    vr : float
        Heliocentric radial velocity of the object (in 100 km/s).
    pml : float
        Proper motion on the galactic longitude (in mas/year).
    pmb : float
        Proper motion on the galactic latitude (in mas/year).
    t0 : float
        Starting time of calculation (in units of 10^7 years).
    tf : float
        Final time of calculation (in units of 10^7 years).
    M_disc : float, default 100
        Mass of the disc, in Msun*10^9. By default, 10^11 Msun.
    M_sph : float, default 30
        Mass of the spherical component of the Galaxy, in Msun*10^9. By default, 3*10^10 Msun.
    rtol : float, default 1e-12
        Relative value of the error of the numerical integration scheme.
    atol : float, default 1e-12
        Absolute value  of the error of the numerical integration scheme.
    name : str, default None
        The name of the cluster.
        By default, name is assigned as "L{l}_B{b}_{direction}", where direction is "forw" or "backw", according to direction of time.
    reverse : bool, default False
        If True, set backward direction of time. By default, direction is forward.
    plot : bool, default False
        Set True to show plots.
    output : str, default None
        The name of the cluster file "{output}_BAR.dat".
        By default, file name is "L{l}_B{b}_{direction}_BAR.dat", where direction is "forw" or "backw", according to direction of the time.
        The first line of the file contains list of input parameters, the second line 
        contains list of columns, see the description of returning parameter.
        If False, do not write the output file.
    return : bool, default False
        If True, 

    Returns
    -------
    data : pandas.DataFrame
        The output DataFrame contains the following columns:
        1) t - time coordinate (in units of 10^7 years), 2) R - distance from the galactic axis (in kpc), 
        3) Vr - dR/dt, radial component of the velocity (in 100 km/s), 4) fi - the position angle relative to Sun direction, counting clockwise 
        if seen from North Galactic Pole (in radians), 5) Vfi - R*d(fi)/dt, tangential velocity (in 100 km/s), 6) z - vertical distance from 
        the galactic plane (in kpc), 7) Vz - dz/dt, vertical velocity (in 100 km/s), 8) E - total energy (in (100 km/s)^2), 9) C - angular 
        momentum (in 100 kpc*km/s), 10) xg - R*cos(fi), X galactocentric coordinates (in kpc), 11) yg - R*sin(fi), Y galactocentric coordinates 
        (in kpc).
    """
    print(
        'Rh = {} kpc, L = {} deg, B = {} deg, Vr = {} km/s, PML = {} mas/yr, PMB = {} mas/yr, T0 = {} Gyr, Tf = {} Gyr'
        .format(rh, lon, lat, float(vr)*100, pml, pmb, float(t0)/100, float(tf)/100)
        )

    if reverse:
        direct = 'back'
        print('Direction: Backward')
    else:
        direct = 'forw'
        print('Direction: Forward')
    if plot:
        print('Plot results: ON')
    else:
        print('Plot results: OFF')
    if output is None:
        out_f_name = 'L' + str(lon) + '_B' + str(lon) + '_' + direct + '_BAR.dat'
        print('Output file: {}'.format(out_f_name))
    elif output is False:
        print('Without outpuf file')
    else:
        out_f_name = output + '_BAR.dat'
        print('Output file: {}'.format(out_f_name))
    if name is None:
        clus_name = 'L' + str(lon) + '_B' + str(lon) + '_' + direct
        print('Cluster name: {}'.format(clus_name))
    elif name is False:
        clus_name = 'L' + str(lon) + '_B' + str(lon) + '_' + direct
        print('Cluster name: {}'.format(clus_name))
    else:
        clus_name = name
        print('Cluster name: {}'.format(clus_name))

    # Global Parametres
    a=5 
    b=0.26
    c=0.7
    d=12
    G = 4.4993154e-10 
    # Vd=6.5593 
    Vd = np.sqrt(M_disc*10**9*G)
    # Vs=3.8247
    Vs = np.sqrt(M_sph*10**9*G)
    Vh=1.15
    Vd2=Vd**2 
    Vs2=Vs**2 
    Vh2=Vh**2
    a_b = 4
    c_b = 1
    e2 = a_b**2 - c_b**2
    eps = np.sqrt(e2)
    Mb = 1e10
    
    betta = 105*Mb*G/(32*eps)
    Fib0 = 45*np.pi/180   
    OmegB = 0.511
    X0 = 1e-7

    def FiBar(R, f, z):
        m = np.sqrt((R*eps*np.cos(f))**2/a_b**2+((R*eps*np.sin(f))**2+(z*eps)**2)/c_b**2)
        if m <= 1:
            psi = np.arccos(c_b/a_b)
        else:
            Ac = -(R*eps*np.cos(f))**2
            Bc = (R*eps)**2 + (z*eps)**2 + e2
            Cc = -e2
            Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
            R1 = (-Bc+Dc)/(2*Ac)
            psi = np.arcsin(np.sqrt(R1))

        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        W10 = 2*np.arctanh(spsi)
        W20 = spsi/cpsi**2 - W10/2
        W11 = W10 - 2*spsi
        W30 = spsi*(1/(2*cpsi**4) - 5/(4*cpsi**2)) + 3*W10/8
        W21 = spsi*(2 + 1/cpsi**2) - 3*W10/2
        W12 = W10 + np.sin(3*psi)/6 - 5*spsi/2
        W40 = 2*spsi*(1/(6*cpsi**6)-13/(24*cpsi**4)+11/(16*cpsi**2)) - 5*W10/16
        W31 = spsi*(1/(2*cpsi**4)-9/(4*cpsi**2)-2) + 15*W10/8
        W22 = spsi*(1/cpsi**2 + 9/2) - np.sin(3*psi)/6 - 5*W10/2
        W13 = W10 - 11*spsi/4 + 7*np.sin(3*psi)/24 - np.sin(5*psi)/40
        sfi = np.sin(f)
        cfi = np.cos(f)  
        return betta*(W10/3 - ((R**2*sfi**2+z**2)*W20 + R**2*cfi**2*W11) +
        ((R**2*sfi**2+z**2)**2*W30 + 2*(R**2*sfi**2+z**2)*R**2*cfi**2*W21 + R**4*cfi**4*W12) -
        (1/3)*((R**2*sfi**2+z**2)**3*W40 + 3*(R**2*sfi**2+z**2)**2*R**2*cfi**2*W31 +
        3*(R**2*sfi**2+z**2)*R**4*cfi**4*W22 + R**6*cfi**6*W13))

    def dFbdR(R, f, z):
        m = np.sqrt((R*eps*np.cos(f))**2/a_b**2+((R*eps*np.sin(f))**2+(z*eps)**2)/c_b**2)
        if m <= 1:
            psi = np.arccos(c_b/a_b)
            dPsidR = 0

        else:
            Bc = (R*eps)**2 + (z*eps)**2 + e2
            Cc = -e2
            if np.abs(R*np.cos(f)) < X0:
                Ac = -X0**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt(-Cc/Bc)
                psi = np.arcsin(hi)
                dAdR = -2*X0**2/R
                dBdR = 2*R*e2
                dHidR = ((-dBdR+(2*Bc*dBdR-4*Cc*dAdR)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdR)/(8*hi*Ac**2)
            else:
                Ac = -(R*eps*np.cos(f))**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt((-Bc+Dc)/(2*Ac))
                psi = np.arcsin(hi)
                dAdR = -2*R*(eps*np.cos(f))**2
                dBdR = 2*R*e2
                dHidR = ((-dBdR+(2*Bc*dBdR-4*Cc*dAdR)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdR)/(8*hi*Ac**2)

            dPsidR = dHidR/np.sqrt(1-hi**2)

        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        W10 = 2*np.arctanh(spsi)
        W20 = spsi/cpsi**2 - W10/2
        W11 = W10 - 2*spsi
        W30 = spsi*(1/(2*cpsi**4) - 5/(4*cpsi**2)) + 3*W10/8
        W21 = spsi*(2 + 1/cpsi**2) - 3*W10/2
        W12 = W10 + np.sin(3*psi)/6 - 5*spsi/2
        W40 = 2*spsi*(1/(6*cpsi**6)-13/(24*cpsi**4)+11/(16*cpsi**2)) - 5*W10/16
        W31 = spsi*(1/(2*cpsi**4)-9/(4*cpsi**2)-2) + 15*W10/8
        W22 = spsi*(1/cpsi**2 + 9/2) - np.sin(3*psi)/6 - 5*W10/2
        W13 = W10 - 11*spsi/4 + 7*np.sin(3*psi)/24 - np.sin(5*psi)/40
        
        dW10dR = 2*dPsidR/cpsi
        dW20dR = 2*dPsidR*spsi**2/cpsi**3
        dW11dR = 2*dPsidR*spsi**2/cpsi
        dW30dR = 2*dPsidR*spsi**4/cpsi**5
        dW21dR = 2*dPsidR*spsi**4/cpsi**3
        dW12dR = 2*dPsidR*spsi**4/cpsi
        dW40dR = 2*dPsidR*spsi**6/cpsi**7
        dW31dR = 2*dPsidR*spsi**6/cpsi**5
        dW22dR = 2*dPsidR*spsi**6/cpsi**3
        dW13dR = 2*dPsidR*spsi**6/cpsi
        
        sfi = np.sin(f)
        cfi = np.cos(f)    
        
        return betta*(dW10dR/3 - ((R**2*sfi**2+z**2)*dW20dR + 2*R*sfi**2*W20 + R**2*cfi**2*dW11dR + 
        2*R*cfi**2*W11) + ((R**2*sfi**2+z**2)**2*dW30dR + 4*(R**2*sfi**2+z**2)*R*sfi**2*W30 +
        2*(R**4*sfi**2*cfi**2+z**2*R**2*cfi**2)*dW21dR + 2*(4*R**3*sfi**2*cfi**2+2*R*z**2*cfi**2)*W21 +
        R**4*cfi**4*dW12dR + 4*R**3*cfi**4*W12) - ((R**2*sfi**2+z**2)**3*dW40dR + 6*(R**2*sfi**2+z**2)**2*R*sfi**2*W40 +
        3*(R**2*sfi**2+z**2)**2*R**2*cfi**2*dW31dR + 12*(R**2*sfi**2+z**2)*R**3*sfi**2*cfi**2*W31 +
        6*(R**2*sfi**2+z**2)**2*R*cfi**2*W31 + 3*(R**6*sfi**2*cfi**4+z**2*R**4*cfi**4)*dW22dR +
        3*(6*R**5*sfi**2*cfi**4 + 4*R**3*z**2*cfi**4)*W22 + R**6*cfi**6*dW13dR +6*R**5*cfi**6*W13)/3)
            
    def dFbdfi(R, f, z):
        m = np.sqrt((R*eps*np.cos(f))**2/a_b**2+((R*eps*np.sin(f))**2+(z*eps)**2)/c_b**2)
        if m <= 1:
            psi = np.arccos(c_b/a_b)
            dPsidf = 0

        else:
            Bc = (R*eps)**2 + (z*eps)**2 + e2
            Cc = -e2
            if np.abs(R*np.cos(f)) < X0:
                Ac = -X0**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt(-Cc/Bc)
                psi = np.arcsin(hi)
                dAdf = 2*(R*eps)*np.sin(f)*X0
                dBdf = 0
                dHidf = ((-dBdf+(2*Bc*dBdf-4*Cc*dAdf)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdf)/(8*hi*Ac**2)
            else:
                Ac = -(R*eps*np.cos(f))**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt((-Bc+Dc)/(2*Ac))
                psi = np.arcsin(hi)
                dAdf = 2*(R*eps)**2*np.cos(f)*np.sin(f)
                dBdf = 0
                dHidf = ((-dBdf+(2*Bc*dBdf-4*Cc*dAdf)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdf)/(8*hi*Ac**2)
                
            dPsidf = dHidf/np.sqrt(1-hi**2)
        
        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        W10 = 2*np.arctanh(spsi)
        W20 = spsi/cpsi**2 - W10/2
        W11 = W10 - 2*spsi
        W30 = spsi*(1/(2*cpsi**4) - 5/(4*cpsi**2)) + 3*W10/8
        W21 = spsi*(2 + 1/cpsi**2) - 3*W10/2
        W12 = W10 + np.sin(3*psi)/6 - 5*spsi/2
        W40 = 2*spsi*(1/(6*cpsi**6)-13/(24*cpsi**4)+11/(16*cpsi**2)) - 5*W10/16
        W31 = spsi*(1/(2*cpsi**4)-9/(4*cpsi**2)-2) + 15*W10/8
        W22 = spsi*(1/cpsi**2 + 9/2) - np.sin(3*psi)/6 - 5*W10/2
        W13 = W10 - 11*spsi/4 + 7*np.sin(3*psi)/24 - np.sin(5*psi)/40
        
        dW10df = 2*dPsidf/cpsi
        dW20df = 2*dPsidf*spsi**2/cpsi**3
        dW11df = 2*dPsidf*spsi**2/cpsi
        dW30df = 2*dPsidf*spsi**4/cpsi**5
        dW21df = 2*dPsidf*spsi**4/cpsi**3
        dW12df = 2*dPsidf*spsi**4/cpsi
        dW40df = 2*dPsidf*spsi**6/cpsi**7
        dW31df = 2*dPsidf*spsi**6/cpsi**5
        dW22df = 2*dPsidf*spsi**6/cpsi**3
        dW13df = 2*dPsidf*spsi**6/cpsi
        
        sfi = np.sin(f)
        cfi = np.cos(f) 
        
        return betta*(dW10df/3 - ((R**2*sfi**2+z**2)*dW20df + 2*R**2*sfi*cfi*W20 +
        R**2*cfi**2*dW11df - 2*R**2*cfi*sfi*W11) + ((R**2*sfi**2+z**2)**2*dW30df +
        4*(R**2*sfi**2+z**2)*R**2*sfi*cfi*W30 + 2*(R**4*sfi**2*cfi**2+z**2*R**2*cfi**2)*dW21df +
        2*(2*R**4*sfi*cfi**3 - 2*R**4*sfi**3*cfi - 2*z**2*R**2*cfi*sfi)*W21 + R**4*cfi**4*dW12df -
        4*R**4*cfi**3*sfi*W12) - ((R**2*sfi**2+z**2)**3*dW40df + 6*(R**2*sfi**2+z**2)**2*R**2*sfi*cfi*W40 +
        3*(R**2*sfi**2+z**2)**2*R**2*cfi**2*dW31df + 12*(R**2*sfi**2+z**2)*R**4*sfi*cfi**3*W31 -
        6*(R**2*sfi**2+z**2)**2*R**2*cfi*sfi*W31 + 3*(R**6*sfi**2*cfi**4+z**2*R**4*cfi**4)*dW22df +
        3*(2*R**6*sfi*cfi**5 - 4*R**6*sfi**3*cfi**3 - 4*z**2*R**4*cfi**3*sfi)*W22 -
        6*R**6*cfi**5*sfi*W13 + R**6*cfi**6*dW13df)/3)    
        
    def dFbdz(R, f, z):
        m = np.sqrt((R*eps*np.cos(f))**2/a_b**2+((R*eps*np.sin(f))**2+(z*eps)**2)/c_b**2)
        if m <= 1:
            psi = np.arccos(c_b/a_b)
            dPsidz = 0
        else:
            Bc = (R*eps)**2 + (z*eps)**2 + e2
            Cc = -e2
            if np.abs(R*np.cos(f)) < X0:
                Ac = -X0**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt(-Cc/Bc)
                psi = np.arcsin(hi)
                dAdz = 0
                dBdz = 2*z*eps**2
                dHidz = ((-dBdz+(2*Bc*dBdz-4*Cc*dAdz)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdz)/(8*hi*Ac**2)
            else:
                Ac = -(R*eps*np.cos(f))**2
                Dc = np.sqrt(Bc**2 - 4*Ac*Cc)
                hi = np.sqrt((-Bc+Dc)/(2*Ac))
                psi = np.arcsin(hi)
                dAdz = 0
                dBdz = 2*z*eps**2
                dHidz = ((-dBdz+(2*Bc*dBdz-4*Cc*dAdz)/(2*Dc))*2*Ac - (-Bc+Dc)*2*dAdz)/(8*hi*Ac**2)

            dPsidz = dHidz/np.sqrt(1-hi**2)
        
        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        W10 = 2*np.arctanh(spsi)
        W20 = spsi/cpsi**2 - W10/2
        W30 = spsi*(1/(2*cpsi**4) - 5/(4*cpsi**2)) + 3*W10/8
        W21 = spsi*(2 + 1/cpsi**2) - 3*W10/2
        W40 = 2*spsi*(1/(6*cpsi**6)-13/(24*cpsi**4)+11/(16*cpsi**2)) - 5*W10/16
        W31 = spsi*(1/(2*cpsi**4)-9/(4*cpsi**2)-2) + 15*W10/8
        W22 = spsi*(1/cpsi**2 + 9/2) - np.sin(3*psi)/6 - 5*W10/2
        
        dW10dz = 2*dPsidz/cpsi
        dW20dz = 2*dPsidz*spsi**2/cpsi**3
        dW11dz = 2*dPsidz*spsi**2/cpsi
        dW30dz = 2*dPsidz*spsi**4/cpsi**5
        dW21dz = 2*dPsidz*spsi**4/cpsi**3
        dW12dz = 2*dPsidz*spsi**4/cpsi
        dW40dz = 2*dPsidz*spsi**6/cpsi**7
        dW31dz = 2*dPsidz*spsi**6/cpsi**5
        dW22dz = 2*dPsidz*spsi**6/cpsi**3
        dW13dz = 2*dPsidz*spsi**6/cpsi
        
        sfi = np.sin(f)
        cfi = np.cos(f)
        
        return betta*(dW10dz/3 - ((R**2*sfi**2+z**2)*dW20dz + 2*z*W20 + R**2*cfi**2*dW11dz) +
        ((R**2*sfi**2+z**2)**2*dW30dz + 4*(R**2*sfi**2+z**2)*z*W30 + 2*(R**4*sfi**2*cfi**2+z**2*R**2*cfi**2)*dW21dz +
        4*z*R**2*cfi**2*W21 + R**4*cfi**4*dW12dz) - ((R**2*sfi**2+z**2)**3*dW40dz +
        6*(R**2*sfi**2+z**2)**2*z*W40 + 3*(R**2*sfi**2+z**2)**2*R**2*cfi**2*dW31dz +
        12*(R**2*sfi**2+z**2)*z*R**2*cfi**2*W31 + 3*(R**6*sfi**2*cfi**4+z**2*R**4*cfi**4)*dW22dz +
        6*z*R**4*cfi**4*W22 + R**6*cfi**6*dW13dz)/3)

    orbit = []
    def sol_f(t, y):
        orbit.append((t, y.copy()))

    #right part for ode-solver
    def galorb(t, y):
        zb=np.sqrt(b**2+y[4]**2)
        rd=np.sqrt(y[0]**2+(a+zb)**2)
        r=np.sqrt(y[0]**2+y[4]**2)
        dFdR=-y[0]*(Vd2/rd**3+Vs2/r/(r+c)**2+2*Vh2/(r**2+d**2))
        dFdz=-y[4]*(Vd2*(1+a/zb)/rd**3+Vs2/r/(r+c)**2+2*Vh2/(r**2+d**2))
        # print(t)
        return np.matrix([[y[1]], [y[3]**2/y[0]+dFdR + dFbdR(y[0]/eps, y[2] + Fib0 + OmegB*t, y[4]/eps)/eps], [y[3]/y[0]],
                        [-y[1]*y[3]/y[0] + dFbdfi(y[0]/eps, y[2] + Fib0 + OmegB*t, y[4]/eps)/y[0]], [y[5]], [dFdz + dFbdz(y[0]/eps, y[2] + Fib0 + OmegB*t, y[4]/eps)/eps]])
                        
    rh = float(rh)
    L = float(lon)*np.pi/180
    B = float(lat)*np.pi/180
    VL=4.741*rh*float(pml)/100
    VB=4.741*rh*float(pmb)/100
    vloc = np.matrix([[float(vr)], [VL], [VB]])

    # Sun's galactocentric distance (in kpc)
    R0=8.3
    # Sun's height (in kpc)
    z0=0.02;
    # Sun's velocity in the galactoicentric frame (in 100 km/s):
    uo=-0.1;
    vo=0.12+2.35;
    wo=0.07;
    # Column Sun velocity vector:
    Velo=np.matrix([[uo], [vo], [wo]])
    R=np.sqrt(rh**2*(np.cos(B)**2)+R0**2-2*R0*rh*np.cos(B)*np.cos(L))

    cosfi=(R0-rh*np.cos(B)*np.cos(L))/R;
    sinfi=rh*np.cos(B)*np.sin(L)/R;
    # Galactocentric rectangular frame to the galactocentric cylindrical frame
    Gg = np.matrix([[cosfi, sinfi, 0], [-sinfi, cosfi, 0], [0, 0, 1]])
    # Local rectangular frame to the local [VR VL VB] frame
    G = np.matrix([[-np.cos(B)*np.cos(L), np.cos(B)*np.sin(L), np.sin(B)],
                [np.sin(L), np.cos(L), 0], [np.sin(B)*np.cos(L),
                -np.sin(B)*np.sin(L), np.cos(B)]])
                
    # Initial coordinates and velocities of the object (cylindrical frame)
    if sinfi>=0:
        fi=np.arccos(cosfi)
    else:
        fi=-np.arccos(cosfi)
    z=z0+rh*np.sin(B)

    uvw0 = Gg*(G.transpose())*vloc + Gg*Velo

    InCond = np.hstack((np.matrix([[R], [fi], [z]]), uvw0))

    if direct == 'back':
        InCond = InCond*np.matrix([[1, 0], [0, -1]])

    u = uvw0[0]
    v = uvw0[1]
    w = uvw0[2]

    if direct == 'back':
        u=-u
        v=-v
        w=-w
        OmegB=-OmegB

    y0 = np.array([R, u, fi, v, z, w], dtype = object)

    solution = ode(galorb).set_integrator('dopri5', rtol = rtol, atol = atol, nsteps = 10000000)
    solution.set_initial_value(y0, float(t0))
    solution.set_solout(sol_f)
    Out = solution.integrate(float(tf))

    sol_t = []
    sol_R = []
    sol_Vrad = []
    sol_fi = []
    sol_Vfi = []
    sol_z = []
    sol_Vz = []

    sol_Fd = []
    sol_Fs = []
    sol_Fh = []
    sol_V2 = []
    sol_E = []   #Energy
    sol_C = []   #Angular momentum

    #Galactical coordinates:
    xg = []
    yg = []
    zg = sol_z

    Fd0 = Vd2/np.sqrt(300**2+(a+b)**2)
    Fs0 = Vs2/(300+c)
    Fh0 = -Vh2*np.log(300**2+d**2)
    Ftot0 = Fd0 + Fs0 + Fh0
    N = len(orbit)
    for j in range(N):
        sol_t.append(orbit[j][0])
        sol_R.append(orbit[j][1][0])
        sol_Vrad.append(orbit[j][1][1])
        sol_fi.append(orbit[j][1][2])
        sol_Vfi.append(orbit[j][1][3])
        sol_z.append(orbit[j][1][4])
        sol_Vz.append(orbit[j][1][5])
        sol_Fd.append(Vd2/np.sqrt(sol_R[j]**2+(a+np.sqrt(sol_z[j]**2+b**2))**2))
        sol_Fs.append(Vs2/(np.sqrt(sol_R[j]**2+sol_z[j]**2)+c))
        sol_Fh.append(-Vh2*np.log(sol_R[j]**2+sol_z[j]**2+d**2))
        sol_V2.append(sol_Vrad[j]**2+sol_Vfi[j]**2+sol_Vz[j]**2)
        sol_E.append(sol_V2[j]/2-(sol_Fd[j]+sol_Fs[j]+sol_Fh[j] - Ftot0 + FiBar(sol_R[j]/eps, sol_fi[j]+Fib0+OmegB*sol_t[j], sol_z[j]/eps)))
        sol_C.append(-sol_R[j]*sol_Vfi[j])
        xg.append(sol_R[j]*np.cos(sol_fi[j]))
        yg.append(sol_R[j]*np.sin(sol_fi[j]))

    data = Table([sol_t, sol_R, sol_Vrad, sol_fi, sol_Vfi, sol_z, sol_Vz, sol_E, sol_C, xg, yg],
                        names = ['t', 'R', 'Vr', 'fi', 'Vfi', 'z', 'Vz', 'E', 'C', 'xg', 'yg'])

    if output is not False:
        
        data.meta['comments'] = ['Rh, L, B, VR, PML, PMB, T0, Tf, Direction = '+direct]
        ascii.write(data, out_f_name, formats = {'t': '%.15f', 'R': '%.15f',
        'Vr': '%.15f', 'fi': '%.15f', 'Vfi': '%.15f', 'z': '%.15f', 'Vz': '%.15f', 'E': '%.15f',
        'C': '%.15f', 'xg': '%.15f', 'yg': '%.15f'})
            
    if plot:
        
        plt.figure(2)
        plt.plot(xg, yg)
        plt.grid(True)
        plt.axis('square')
        plt.xlabel('X, kpc')
        plt.ylabel('Y, kpc')
        plt.title('Orbit projection onto the galactic plane')
        plt.savefig(clus_name + '_gal_plane_BAR.png')
        
        plt.figure(3)
        plt.plot(sol_R, zg)
        plt.grid(True)
        plt.xlabel('R, kpc')
        plt.ylabel('Z, kpc')
        plt.title('Meridional section of the orbit')
        plt.savefig(clus_name + '_merid_sec_BAR.png')
        
        fig = plt.figure(4)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xg, yg, zg, 'b-')
        ax.plot([R0], [0], [z0], 'yo', markersize=10)
        ax.plot([0], [0], [0], 'g*', markersize=10)
        ax.set_xlabel('X, kpc')
        ax.set_ylabel('Y, kpc')
        ax.set_zlabel('Z, kpc')
        ax.set_title('Orbit in 3D')
        plt.savefig(clus_name + '_3D_BAR.png')

    return data.to_pandas()