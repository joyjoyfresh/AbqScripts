# The combined amplification effects of topography and stratigraphy of layered rock slopes under vertically and obliquely incident seismic waves

![](images/1ac5454202bc5399c8b7d7bdf787571e1451615d8105ebaa42acd6ed53eb7964.jpg)

Hui Shen <sup>a</sup> , Yaqun Liu <sup>c,\*</sup> , Xinping Li <sup>b,a</sup> , Haibo Li <sup>c</sup> , Liangjun Wang <sup>a,d</sup> , Wenxu Huang <sup>b,a</sup>

<sup>a</sup> School of Civil Engineering and Architecture, Wuhan University of Technology, Wuhan, Hubei, 430070, China  
<sup>b</sup> Sanya Science and Education Innovation Park of Wuhan University of Technology, Sanya, Hainan, 572025, China  
<sup>c</sup> Institute of Rock and Soil Mechanics, Chinese Academy of Sciences, Wuhan, Hubei, 430071, China  
<sup>d</sup> China Gezhouba Group Co.,Ltd., Wuhan, Hubei, 430070, China

## A R T I C L E I N F O

Keywords:

Rock slope

Dynamic response

Wave propagation

Seismic amplification

Oblique incidence

Spectral element method

## A B S T R A C T

Both the topography and stratigraphy of slopes significantly affect the ground motions of slopes during earthquakes, and oblique incidence of seismic waves can further aggravate amplification. This study aims to parametrically explore the combined effects of the topography and stratigraphy of layered rock slopes on seismic amplification subjected to vertical and oblique propagating waves and provide qualitative and quantitative insight into this phenomenon. The spectral element method used to obtain the seismic response of slopes is introduced and verified by two examples. The influences of the slope angle, material properties of the layers, surface layer conditions, and incident angle of the seismic waves on the seismic amplification are then investigated. The results indicate that the peak horizontal and vertical amplification factors for layered rock slopes subjected to vertical and oblique incidence of seismic waves are in the ranges of 1.3–7.6 and 0.3–5.2, respectively. Among the various factors, the thickness and shear wave velocity of the surface layer of slopes have the greatest influence on the amplification effect, especially for obliquely incident waves. At oblique incidence, the maximum horizontal and vertical normalized acceleration amplification factors for the soft-surface-layer slope are 4.4 and 7.4 times greater than those for the hard-surface cases, respectively, whereas at vertical incidence, these values are only 2.8 and 4.3, respectively. When the impedance ratio between the surface layer and the underlying layer is 0.5 (i.e., the soft surface layer), unusual vertical amplification is observed where the maximum vertical amplification factor reaches 5.2. The findings of this study may provide useful reference and guidance for the seismic design of slope engineering and building structures near slopes.

## 1. Introduction

Earthquakes are among the most devastating natural disasters, particularly in mountainous areas. Secondary disasters such as landslides, rockfalls, and debris flows triggered by earthquakes have caused thousands of casualties and countless property losses [1–4]. Numerous post-earthquake investigations and studies have demonstrated that irregular topographic features, such as slopes, ridges, cliffs, and isolated hilltops, can significantly amplify seismic ground motions, leading to more severe earthquake damage [5–8]. Consequently, comprehending and quantifying the amplification effect of slopes can provide valuable insights and guidance for the seismic design of slope engineering.

The methods used to study the amplification effect of slopes can generally be divided into three main categories: field observations [9–11], shaking table tests [12–14], and numerical simulations [15–17]. The recorded seismic data are beneficial for field investigations, but they are limited by the random nature of seismic events. Shaking table tests can be used to reconstruct seismic events and capture the authentic dynamic response of slope models during earthquakes, but these tests are usually time-consuming and economically expensive. Over the past few decades, numerical approaches, including the finite difference method [18,19], finite element method [20], discrete element method [21], and spectral element method [22], have been extensively developed to investigate the amplification effect of slopes. In the analysis of the amplification effects of slopes, most results are based on vertically propagating waves in both physical models and numerical models, which usually ignore the directivity of seismic responses. However, for earthquakes with shallow epicenters, it is essential to consider the impact of the incident angle of seismic waves on ground motion amplification at local sites. The incident angle of seismic waves near the ground surface is usually inclined, as summarized by the statistical analysis of seismological data [23].

Post-earthquake field investigations have shown that the angle of incidence of seismic waves can significantly change the pattern of seismic ground motions, which is crucial for interpreting the seismic response of slopes associated with the amplification effect. A statistical analysis on the slip directions of 85 large-scale landslides triggered by the Wenchuan earthquake indicated that the intensity of ground motion was significantly related to the directionality of seismic waves [24]. A retrospective investigation of the Lorca rockslide event also revealed the important influence of the specific interactions between irregular topography and obliquely propagating seismic waves on triggering coseismic landslides in southeastern Spain [25]. Additionally, analysis of peak ground acceleration (PGA) and spectral data recorded at the Tarzana site during the Northridge earthquake suggested that local topography and geology could not completely explain the observed seismic amplification phenomenon unless it was assumed that the seismic waves propagated in an oblique direction [26].

Slope topography and the propagation direction of seismic waves are both important factors that affect seismic ground motion; thus, many numerical studies have been conducted to investigate topographic site effects under obliquely incident waves. Ashford and Sitar [27] performed a parametric study to evaluate the topographic amplification of inclined shear waves in steep slopes. Their findings indicated that the amplification of obliquely propagating waves can be more than twice the amplification due to vertically propagating waves. Fan et al. [28] analyzed the dynamic response of homogeneous rock slopes under obliquely incident SV-waves via wave field decomposition theory. They reported that the incident angle of seismic waves significantly influenced the dynamic response of rock slopes, which can be underestimated if oblique incidence was neglected. Yin et al. [29] evaluated the impact of incident angles of earthquake SV-waves on a double-faced homogeneous soil slope and revealed that obliquely incident waves altered the directivity of slope seismic responses. Our previous parametric study investigated the time-domain and frequency-domain topographic amplification of homogeneous rock slopes under obliquely incident waves by considering different slope geometric parameters (slope angles and slope heights) [30]. Our results showed that the maximum PGA amplification factors under oblique incident waves were generally 1.4–2.2 times those under vertical incidence. All these studies indicated that the topographic amplification effect under oblique incident waves was more obvious than that under vertical incidence. However, case studies reported by Gallipoli et al. [31] and Hailemikael et al. [32] emphasized that the amplification patterns of slopes were also highly dependent on stratigraphic conditions. Diffraction and interference can occur when seismic waves travel through mountain topography and stratigraphy, thus altering the pattern of localized (de) amplification of ground acceleration [33]. The thickness and geometry of the layer and the strong contrast in impedance between the layer and the underlying bedrock were mainly responsible for stratigraphic site effects [34]. Given the extremely limited number of studies that have considered the influence of the incident angle of seismic waves on the combined amplification effects of the stratigraphy and topography of rock slopes, a systematic parametric investigation is needed to further quantitatively reveal the impact of the interaction between oblique seismic waves, slope topography and stratigraphy on seismic amplification.

The main goal of this study is to advance the understanding of the combined amplification effects of the topography and stratigraphy of layered slopes subjected to vertical and oblique seismic waves. SPEC-FEM2D, an open-source software package based on the spectral element method theory, was employed to model the seismic response of rock slopes. First, the SPECFEM2D was modified to meet the needs of specific load requirements, and two test examples were used to verify the accuracy and efficiency of the modified software. A systematic parametric study was then performed by considering different slope geometries, stratigraphic conditions, and incident angles of seismic waves. The effects of the slope angle, depth-to-bedrock ratio, impedance ratio between different layers, and surface layer properties on the ground motion amplification of layered rock slopes subjected to vertical and oblique incidences of SV-waves were quantitatively investigated. Finally, the mechanism of the earthquake‒slope interactions was analyzed, and a comparison of the amplification factors between the seismic codes and numerical results was performed.

## 2. Method and model setup

## 2.1. Spectral element method

As mentioned previously, numerical methods have been well developed and extensively applied to study the topographic and stratigraphic effects of irregular features under complex conditions [35,36]. The spectral element method (SEM), which combines the accuracy of the pseudospectral method with the flexibility of the finite element method, is currently one of the most widely used numerical approaches for seismic wave modeling [37,38]. The open-source software package SPECFEM2D [39,40], developed on the basis of SEM theory, is a powerful 2D spectral-element solver. It is mainly used for simulating seismic wave propagation and full waveform imaging or adjoint tomography [41]. SPECFEM2D accommodates both absorbing boundary conditions and higher-order time schemes, which offers the advantages of good accuracy and convergence properties. Consequently, SPEC-FEM2D is employed to analyze the seismic response of rock slopes under obliquely incident earthquake-SV waves in this study.

## 2.2. Modeling oblique incident waves

In SPECFEM2D, the model information files, simulation parameter files, and seismic source files are separate. The slope model in this study is created and meshed in third-party software (e.g., Gmsh). When creating the slope model, absorbing boundary flags were applied to the left, right, and bottom truncated boundaries, and a free boundary flag was applied to the top surface of the model. According to Kuhlemeyer and Lysmer [42], the maximum element size Δl must be smaller than approximately one-tenth of the wavelength associated with the highest frequency component of the input wave to avoid numerical distortion,

$$
(\Delta l) _ {m a x} \leq \frac {(c _ {s}) _ {m i n}}{1 0 f _ {m a x}} \tag {1}
$$

Where $( c _ { s } ) _ { m i n }$ is the minimum shear velocity in the model and the $f _ { m a x }$ is the highest frequency component of the input wave. Then, the node and element information for discrete domains, absorbing boundary and free surface information for dynamic boundary conditions, and partitioning information for different materials are converted into a series of model input files.

To simulate the propagation of incoming plane waves in the domain, it is necessary to set initial plane wave conditions in SPECFEM2D. More specifically, the incident condition for the plane wave is specified by setting the “initialfield” parameter and the “Bielak” conditions in the parameter file. Prior to the dynamic analysis, the absorbing boundary condition (ABC) should be specified in the left, right, and bottom of the model to dissipate the reflected wave energy at the truncated boundary. Note that for plane wave incidence, the Stacey ABC must be used in SPECFEM2D instead of the commonly used Perfectly Matched Layer (PML) boundary. In addition, the time step should be appropriately selected according to the Courant–Friedrichs–Lewy (CFL) condition to ensure numerical stability. The material parameters for the model, the receiver parameters for recording stations (i.e. recording points), and output parameters for visualization and analysis should also be specified in the parameter file. Then the seismic response of the model can be obtained by the forward simulation of seismic wave propagation.

This study involved modifying portions of the Fortran source code of SPECFEM2D and recompiling the software to accommodate the load requirements of numerical simulation. More specifically, we modified the default Ricker displacement source to a Ricker acceleration source in the Fortran code files associated with computing the analytical initial plane wave and added an amplitude factor to scale it. The source code and compilation process of the SPECFEM2D software can be found on the GitHub website (https://github.com/SPECFEM/specfem2d). Therefore, it is essential to verify the correctness and accuracy of the oblique incidence of plane SV-waves in the modified SPECFEM2D.

## 2.3. Verification

## 2.3.1. Test example of elastic half-space

The seismic response of a rectangular foundation was calculated to verify the applicability of modified SPECFEM2D in simulating wave propagation in an elastic half-space. As shown in Fig. 1(a), the total height and length of the numerical model were 2000 m and 1000 m, respectively. The model had a density of 2500 $\mathbf { k g } / \mathbf { m } ^ { 3 } ;$ , a Poisson’s ratio of 0.3, and an elastic modulus of 20 GPa. Three observation points (marked with red circles) were located on the free ground surface, and one was located underground. As illustrated in Fig. 1(b), a Ricker wavelet with an amplitude of 1 $\mathbf { m } / \mathbf { s } ^ { 2 }$ and a center frequency of $f _ { c } = 5$ Hz was utilized as the incident SV-wave. The incident angle of the seismic wave (θ<sub>s</sub>) was defined as the angle between the vertical direction and the direction of wave propagation. The incident angle was set to 15<sup>◦</sup> in this example. The maximum element size of the model was determined as $5 \mathrm { m } ,$ which was small enough to avoid numerical distortion of the propagating wave. The time step was $5 \times 1 0 ^ { - 5 }$ s, which satisfied the CFL condition. The total simulation time was 2.0 s in this simulation. Fig. 1(c) shows snapshots of the displacement of the half-space under the Ricker SVwave with an incident angle of 15<sup>◦</sup> . This figure clearly shows the incidence and propagation of the SV-wave from the lower left corner of the model and its decomposition after it reaches the free ground surface.

The theoretical values of the displacements of the observation points can be obtained on the basis of wavefield decomposition theory. For a given node in the elastic half-space model, the displacement solution is as follows [43]:

$$
\left\{ \begin{array}{c} \mathrm{U} _ {x} (t) = u _ {0} (t - \Delta t _ {1}) \cos \alpha - A _ {1} u _ {0} (t - \Delta t _ {2}) \cos \alpha \\ + A _ {2} u _ {0} (t - \Delta t _ {3}) \sin \beta \\ \mathrm{U} _ {y} (t) = - u _ {0} (t - \Delta t _ {1}) \sin \alpha - A _ {1} u _ {0} (t - \Delta t _ {2}) \sin \alpha \\ - A _ {2} u _ {0} (t - \Delta t _ {3}) \cos \beta \end{array} \right. \tag {2}
$$

where $\mathrm { U } _ { x } ( t )$ and $\mathrm { U } _ { y } ( t )$ are the horizontal and vertical displacements of the node, respectively. $u _ { 0 } ( t )$ represents the displacement of the incident SV-wave. Δt<sub>i</sub> $( i = 1 , 2 , 3 )$ is the travel time of the SV-wave propagating from the wavefront at time zero to the node. α and β represent the angles of incident SV-wave and reflected P-wave, respectively. $A _ { 1 }$ and $A _ { 2 }$ represent the amplitude ratio of the reflected SV-wave and P-wave to the incident SV-wave, respectively.

Fig. 2 compares the analytical solutions of displacement components at the observation points with the numerical results of SPECFEM2D. The numerical results are generally in good agreement with the analytical solutions. Small fluctuations are observed in the numerical solution after

![](images/92cb738cc5a352b00a670ef0f01d67b640921395e5fc512cf81cdaf5330ef98d.jpg)  
Fig. 1. (a) Elastic half-space model with illustrative receiver locations and incident waves, (b) acceleration time history of the input wave, and (c) snapshots of the displacement fields of the model.

![](images/6cb7b449b6b0f598c7985517f9409a7fa6fb923d071221c8bb2969d40f1389fb.jpg)  
Time (s)  
Fig. 2. Comparison between the analytical solutions and numerical simulations of the displacement time history at the observation points.

the peak values, which lead to some deviations between the numerical and analytical solutions. It may be attributed to the adopted Stacey ABC, which was better for absorbing P-wave energy, worse for absorbing Swave energy [44], resulting in the reflected wave energy not being completely dissipated at the truncation boundary and part of the S-wave being reflected. Nonetheless, the comparative results still demonstrated that the modified SPECFEM2D is suitable for efficiently and accurately modeling the propagation of oblique waves in half-space.

## 2.3.2. Test example of a rock slope

To further verify the ability of SPECFEM2D to model the propagation of oblique waves in slopes, we calculated the dynamic response of the same step-like slope model subjected to the oblique incidence of SVwaves using both the SPECFEM2D and the equivalent nodal force method implemented in ABAQUS [45]. The geometry and material parameters of the slope are shown in Fig. 3(a). The slope height was 200 m, and the slope angle was 45<sup>◦</sup> . The density of the model was 2500 kg/m<sup>3</sup> , the elasticity modulus was 20 GPa, and the Poisson’s ratio was 0.3. The incident SV-wave was a Ricker wavelet with an amplitude of 1 $\mathbf { m } / s ^ { 2 } ;$ , a center frequency of 8 Hz, and an angle of incidence of 10<sup>◦</sup> . Note that the artificial viscous-spring boundary was used in the ABAQUS model, which served as absorbing boundary conditions. The maximum element size for both the SPCFEM2D model and the ABAQUS model was 4 m, which was small enough to represent the accurate wave transmission through the model. The time step was $2 \times 1 0 ^ { - 5 }$ s and the total simulation time was 1.0 s in this case.

![](images/87ec8e8ca054083877ee05e1549ddeeb761d43ff0bfc75f19f835765740058ce.jpg)  
(a)  
(b)  
Fig. 3. Displacement contours of the slope under the obliquely incident SV-waves obtained from (a) SPECFEM2D and (b) ABAQUS.

The results of the two methods are compared in terms of displacement field snapshots, as shown in Fig. 3. Two observation points, A and $\mathbf { B } ,$ representing the top and foot of the slope, respectively, are labeled in Fig. 3(a), and the moments of the snapshots are shown in the upper-left corner of each subplot. The displacement fields of the slope obtained by different methods look similar. The results of these snapshots indicate that the absorbing boundaries work well in both ABAQUS and SPEC-FEM2D despite their different theoretical foundations.

The horizontal acceleration (denoted by A1) and vertical acceleration (denoted by A2) components of points A and B obtained via the two methods are depicted in Fig. 4. The numerical results of SPECFEM2D are in good agreement with those of ABAQUS, illustrating that the modified SPECFEM2D can efficiently model the propagation of obliquely incident seismic waves and accurately simulate the corresponding seismic response when analyzing slope topography.

## 2.4. Model setup

## 2.4.1. Model configuration

To investigate the amplification of obliquely incident seismic waves, two simplified layered rock slope models were considered: one consisting of a homogeneous layer over an underlying bedrock layer (Fig. 5 (a)) and the other with a surface layer added to the previous slope model (Fig. 5(b)). The solid interfaces were assumed to be horizontal, and the wave velocities of the materials between these layers were not identical.

![](images/b409ff30ca9cfb2cb93171f35cc14c096730709228b6224beeb3fa76eb66105f.jpg)  
Fig. 4. The variation in acceleration of points A and B with time under the oblique incidence of the SV-wave.

![](images/3bf7139b56ee12e2951798cdcf89ec391c0f1cc916a445b97260916cf4d3a0eb.jpg)

<details>
<summary>text_image</summary>

1000 m
800 m
Upper ground surface
#1
Slope face
200 m
H
Overlying layer [Vs]
Lower ground surface
#2
h
Bedrock [VR]
Absorbing boundary
Incident wave
θs
</details>

(a) Homogeneous Model

![](images/4e793e0bd90a1a0627b0ed1b4f813b464a2072c525d4530385306865e5f5fa24.jpg)

<details>
<summary>text_image</summary>

1000 m
800 m
h₁ Surface layer [Vₛ₁] #1
200 m i Slope face
H Overlying layer [Vₛ₂] #2
h
θₛ Bedrock [Vᵣ] Absorbing boundary
Incident wave
</details>

(b) Two-Layered Model  
Fig. 5. Schematic representation of the numerical models for analyzing the dynamic response of step-like slopes.

The shear wave velocities of different layer materials are indicated in Fig. 5, with the shear wave velocity of the bedrock denoted by $V _ { R }$ and that of the layers overlying the bedrock denoted by $V _ { s } , V _ { s 1 }$ and $V _ { s 2 } .$ . The seismic wave was incident from the bottom-left corner of the slope model, and the incident angle of seismic waves (θ ) was described by the angle between the wave propagation direction and the vertical direction. The total length of the slope model was 1800 ${ \mathfrak { m } } ,$ and the length of the upper ground surface was 1000 m. The slope height was a constant 200 m, but the slope angle varied to represent different topographic features. Additionally, an absorbing boundary condition was adopted for the truncated boundary of the slope model to dissipate the energy of the reflected and scattered waves.

## 2.4.2. Source configuration

The commonly used Ricker wavelet was adopted as the input wave, and two incident angles were considered for each numerical model, including vertical incidence (0<sup>◦</sup> ) and oblique incidence (15<sup>◦</sup> ). The acceleration time history of the SV-Ricker wave is defined as follows:

$$
r (t) = \left(1 - 2 \pi^ {2} f _ {c} ^ {2} (t - t _ {0}) ^ {2}\right) e ^ {\left(- \pi^ {2} f _ {c} ^ {2} (t - t _ {0}) ^ {2}\right)} \tag {3}
$$

where $f _ { c }$ represents the central frequency of the Fourier spectrum and the time. t and $t _ { 0 }$ represent the time and moment corresponding to the peak acceleration, respectively. Fig. 6 shows the typical acceleration time histories of Ricker wavelets with different central frequencies.

## 2.4.3. Simulation design

The following dimensionless parameters are considered and investigated: (1) the dimensionless frequency, which is normalized by the height of the slope and the velocity of the shear wave, i.e., $a _ { 0 } = 2 f _ { c } ( H$ − $h ) / V _ { s ; \thinspace } ( 2 )$ the impedance ratio between the bedrock and an overlying layer, $\rho _ { R } V _ { R } / \rho _ { s } V _ { s } ,$ , and the depth-to-bedrock ratio, h/H; and (3) the relative thickness $( h _ { 1 } / ( H - h ) )$ of the surface layer and the impedance ratio $( \rho _ { s 1 } V _ { s 1 } / \rho _ { s 2 } V _ { s 2 } )$ between the surface layer and the underlying layer.

In the parametric study, the density and Poisson’s ratio of both the upper layers and the underlying bedrock were 2500 ${ \bf k g } / { \bf m } ^ { 3 }$ and $0 . 3 ,$ respectively. The elastic modulus of the bedrock was 26 GPa, and its shear and compression wave velocities were 2000 m/s and 3742 ${ \mathfrak { m } } / { \mathfrak { s } } ,$ respectively. Other physical and mechanical properties of the upper layers are determined by the impedance ratios between the bedrock and the overlying layer or between the surface layer and the overlying layer (Table 1). The density of the upper layer is chosen to be identical to that of the underlying bedrock; therefore, the impedance ratio between the bedrock and overlying layer is simply given by $V _ { R } / V _ { s }$ . Fig. 7 shows the slope geometry configurations and stratigraphic features. For the homogeneous model, three slope angles were selected to represent different topographic features and different impedance ratios $( \rho _ { R } V _ { R } /$ $\rho _ { s } V _ { s } )$ between the bedrock and overlying layer as well as varying depthto-bedrock ratios $( h / H )$ were considered. For the Two-Layered model, the stratigraphic properties, described by the impedance ratio $\left( V _ { R } / \ V _ { s } \right)$ between the bedrock and the overlying layer, the flexibility $\left( V _ { s 1 } / \ V _ { s 2 } \right)$ of a surface layer, and the relative thickness $( h _ { 1 } / ( H - h ) )$ , were taken into account. Except for the physical and mechanical properties of the material, the bulk and shear quality factors $( Q _ { \kappa }$ and $Q _ { \mu } )$ also have been used in the model to represent seismic attenuation. The quality factor (Q) is defined as a measure of the quality of an oscillating system, representing the ratio of stored energy to dissipated energy. Q has an inverse relationship with attenuation, the smaller $Q ,$ the larger the attenuation. In the model, the coarse-grain method is used to calculate the quality factors $( Q _ { p }$ and $Q _ { s } )$ based on the wave velocity of layers, $Q _ { s } = 0 . 0 5 c _ { s } , Q _ { p }$ $= 2 Q _ { s }$ [46]. For 2D plane strain problems in SPECFEM2D, the P-wave and S-wave quality factors $Q _ { p }$ and $Q _ { s }$ are related to the bulk and shear quality factors $Q _ { \kappa }$ and $Q _ { \mu }$ by

![](images/f651b9898a54bbc19b2909017a2b32383fff6815f5e7416da2d1a92b1233b291.jpg)

<details>
<summary>line</summary>

| t (s) | fc = 8 Hz | fc = 4 Hz | fc = 2 Hz | fc = 1 Hz |
| --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ~0.15 | 1.0 | — | — | — |
| ~0.25 | — | 1.0 | — | — |
| ~0.35 | — | — | -0.45 | — |
| ~0.65 | — | — | -0.45 | — |
| ~1.0 | — | — | — | 1.0 |
| ~1.4 | — | — | — | -0.45 |
| 2.0 | 0.0 | 0.0 | 0.0 | 0.0 |
</details>

Fig. 6. Ricker wavelets with different central frequencies.

$$
Q _ {p} ^ {- 1} = \left(1 - \frac {c _ {s} ^ {2}}{c _ {p} ^ {2}}\right) Q _ {k} ^ {- 1} + \left(\frac {c _ {s} ^ {2}}{c _ {p} ^ {2}}\right) Q _ {\mu} ^ {- 1} \tag {4}
$$

$$
Q _ {s} ^ {- 1} = Q _ {\mu} ^ {- 1}
$$

Therefore, the bulk and shear quality factors $( Q _ { \kappa }$ and $Q _ { \mu } )$ of the different layers can be determined. Note that the bedrock has a quality factor of 999 to ignore seismic wave attenuation. Table 1 lists the parameters considered in the numerical simulations.

Table 1 Parameters considered in the numerical simulations.

<table><tr><td colspan="2">Homogeneous Profile</td><td colspan="6">Two-Layered Profile</td></tr><tr><td> $i$ </td><td> $a_0 = 2f_0(H - h)/V_s$ </td><td> $V_R/V_s$ </td><td> $h/H$ </td><td> $i$ </td><td> $V_R/V_{s2}$ </td><td> $V_{s1}/V_{s2}$ </td><td> $h_1/(H - h)$ </td></tr><tr><td> $30^\circ$ </td><td>0.5</td><td>1.25</td><td>0.25</td><td> $45^\circ$ </td><td>1.25</td><td>0.5</td><td>0.25</td></tr><tr><td> $45^\circ$ </td><td>1</td><td>2.5</td><td>0.5</td><td>-</td><td>2.5</td><td>0.75</td><td>0.5</td></tr><tr><td> $60^\circ$ </td><td>1.5</td><td>5.0</td><td>0.75</td><td>-</td><td>-</td><td>2.0</td><td>0.75</td></tr><tr><td>-</td><td>2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.0</td></tr></table>

## 3. Results

## 3.1. Effect of slope angle

Assimaki et al. [9] suggested a topographic amplification factor (TAF) to indicate and quantify the degree of seismic amplification. TAF is a normalized PGA amplification factor, which is defined as the ratio of the PGA at the ground surface to that of the free field:

$$
\mathrm{TAF} _ {h} = \frac {a _ {h , m a x} ^ {x}}{a _ {h , m a x} ^ {f f}}, \mathrm{TAF} _ {\nu} = \frac {a _ {\nu , m a x} ^ {x}}{a _ {h , m a x} ^ {f f}} \tag {5}
$$

Where the subscripts h and v in each term represent the horizontal and vertical directions, respectively. a is the ground acceleration, and the superscripts x and $\mathcal { H }$ represent the horizontal coordinates on the ground surface and free field, respectively.

To quantify the amplification effect on rock slopes, the horizontal and vertical PGA amplification factors are calculated according to $\operatorname { E q . }$ (3). Taking the slope configuration with $V _ { R } / V _ { s } = 1 . 2 5$ and $h / H = 0 . 5$ as an example, Fig. 8 shows the distributions of the horizontal and vertical PGA amplification factors on the ground surface for different slope angles and incident angles of seismic waves. In each subfigure, #1 and #2 represent the locations of the crest and toe of the slope, respectively. The dominant frequency of the incident wave is characterized by the normalized dimensionless frequency parameter $\left( a _ { 0 } \right)$ . The main results of our investigation can be summarized as follows. (1) For a gentle slope (i $= 3 0 ^ { \circ } )$ ), the location of the peak horizontal PGA amplification factors behind the slope crest is controlled by the dominant wavelength of the incident motion under vertically incident waves, which was found to be consistent with those reported by Ashford et al. [47] and Assimaki et al. [9]. Under oblique incidence of seismic waves, the peak horizontal PGA amplification factors mainly occur in the vicinity of the slope crest, which is less frequency-dependent. For a steep slope $( i = 6 0 ^ { \circ } )$ , the oblique incidence effect is significant for both the horizontal and vertical PGA amplification factors. (2) The peak values of the horizontal and vertical PGA amplification factors along the ground surface of a slope generally increase with increasing dimensionless frequency $a _ { 0 } .$ . However, for the slope with an angle of 60<sup>◦</sup> , the peak value of the horizontal PGA amplification factor decreases significantly with increasing dimensionless frequency under the oblique incidence of seismic waves, which indicates that the frequency-dependent nature of wave diffraction by surface topographic features is more obvious for steep slopes under obliquely incident waves. (3) The horizontal amplification effect is considerably greater than the vertical amplification effect. As the slope angle increases, the vertical PGA amplification factors increase but remain less than or approximately 1.0.

Fig. 9 shows the variation in the peak horizontal and vertical PGA amplification factors with respect to the dimensionless frequency for both vertically and obliquely incident waves. The linear trend lines fitted on the basis of the data points are also plotted in the subfigures. Both the horizontal and vertical peak PGA amplification factors along the ground surface clearly increase with increasing dimensionless frequency under the condition of vertical incidence of seismic waves. Moreover, for a given dimensionless frequency, the maximum horizontal PGA amplification factors of the gentle slopes (e.g., $i = 3 0 ^ { \circ } )$ are greater than those of the steep slopes $( \mathbf { e } . g . , i = 6 0 ^ { \circ } ) ,$ whereas the vertical amplification pattern is the opposite. Under the oblique incidence of seismic waves, the peak horizontal PGA amplification factor of the slope surface tends to decrease with increasing dimensionless frequency, while the peak vertical PGA amplification factor increases with the increase of the dimensionless frequency.

To further investigate the mechanism of the above amplification phenomena, snapshots of the acceleration wave field of the slope and synthetic seismograms of the slope surface under obliquely incident waves are plotted in Fig. 10 for slopes with angles of 30<sup>◦</sup> and 60<sup>◦</sup> . In Fig. 10(a) and (b), the magnitude and component of the acceleration wave fields are shown in each column. In ${ \mathrm { F i g . ~ } } 1 0 ( { \mathrm { c } } ) ,$ , the vertical axis of the subfigures represents the location of the receivers (i.e., the observation points) on the ground surface of the slope, and the horizontal axis represents the time. The main results of our investigation can be summarized as follows. (1) The total acceleration wave fields are similar for both cases, whereas the distributions of the vertical acceleration wavefield differ between the gentle slope and steep slope. An obvious surface wave is observed in the upper ground surface of the steep slope, which may explain why the vertical acceleration component is significantly enhanced. (2) Stronger P-waves and Rayleigh waves are generated at both the crest and toe of the slope because of the presence of sharp corners. The steeper slope angles and inclined incident waves further distort the diffracted wave field of the slope, which results in a more complex seismic ground motion response on the slope face.

![](images/fbfdc2a59ffd0b4988a7747d1372f90e67d89a2ca49100d3cb5c263b4c2626bf.jpg)  
Fig. 7. Sketch map of the slope geometry configurations and stratigraphic features.

## 3.2. Effect of the depth-to-bedrock ratio

To investigate the influence of depth-to-bedrock ratio $( h / H )$ on the amplification effect of slopes, rock slope models with different $h / \ H$ values were designed. Taking a slope configuration with $i = 3 0 ^ { \circ }$ and $V _ { R } /$ $V _ { s } = 1 . 2 5$ as an example, Fig. 11 shows the distributions of the horizontal PGA amplification factors of slopes with different depth-tobedrock ratios under vertical incidence and oblique incidence of seismic waves, respectively (the subplots are grouped by dimensionless frequency, varying from 0.5 to 2.0).

Under the vertical incidence of seismic waves (Fig. 11(a)), the results can be summarized as follows. (1) For a given incident wave frequency, the effect of the depth-to-bedrock ratio on the horizontal PGA amplifi cation factor is not significant, and its peak value is almost the same despite different $h / H$ values. (2) The peak value of the horizontal PGA amplification factor increases with increasing dimensionless frequency a . For example, the peak value of the horizontal PGA amplification factor is only approximately 1.4 at $a _ { 0 } = 0 . 5 ,$ , whereas the peak value increases rapidly to more than 1.8 at $a _ { 0 } = 2 . 0$ .

Under the oblique incidence of seismic waves (Fig. 11(b)), the results can be summarized as follows. (1) For a given incident wave frequency, the peak value of the horizontal PGA amplification factor increases with increasing thickness of the overlying layer. (2) The maximum value of the horizontal PGA amplification factor does not change significantly for different incident wave frequencies, and the maximum values are in the range of 1.8–1.9. (3) An increase in the dimensionless frequency also affects the shape of the PGA amplification factor curve and intensifies its oscillation.

To compare more intuitively the influences of the depth-to-bedrock ratio, incident wave frequency, and incident angle of the seismic wave on the amplification effect of the slope, Fig. 12 shows the variations in the maximum values of the horizontal and vertical PGA amplification factors with the dimensionless frequency a under different incident angles $\theta _ { s } .$ . The influence of the depth-to-bedrock ratio on the amplification effect of the slope is relatively small under a vertical incident wave, but the incident wave frequency has a significant influence on it. More specifically, both the horizontal and vertical peak PGA amplification factors increase with increasing dimensionless frequency, and the increase is greater in the horizontal direction (from 1.40 to approximately 1.85) and smaller in the vertical direction (only from 0.35 to approximately 0.45). The influence of the depth-to-bedrock ratio on the amplification effect begins to emerge in the case of obliquely incident seismic waves. For a given incident wave frequency, the peak values of both the horizontal and vertical PGA amplification factors increase as the depth-to-bedrock ratio increases, indicating that the effect of the thickness of the overlying layer on the amplification of seismic ground motion needs to be taken into account for obliquely incident waves. We speculate that obliquely incident seismic waves are more likely to cause surface waves on the slope surface. In addition, as the time and distance of obliquely incident waves traveling inside the slope increase, the interactions among diffracted, reflected, and incident waves become more pronounced, resulting in a complex wave field that leads to stronger motions on the slope surface.

It is also found that the peak value of the horizontal amplification PGA factor does not change much with the increase of the dimensionless frequency, and its value remains in the range of 1.65–1.90 without significant growth. On the other hand, the peak value of the vertical PGA amplification factor of the slope surface tends to decrease with increasing dimensionless frequency, which is contrary to the results of vertically incident seismic waves.

In summary, under a vertically incident wave, the horizontal and vertical amplification effects of the slope are affected mainly by the incident wave frequency. Under an obliquely incident wave, the horizontal amplification effect is affected mainly by the thickness of the overlying layer, whereas the vertical amplification effect is affected mainly by the dominant frequency of the incident wave.

## 3.3. Effect of the impedance ratio

To analyze the influence of the impedance ratio of the bedrock to the overlying layer $( V _ { R } / V _ { s } )$ on the amplification effect of the slope, the slope configuration with $i = 4 5 ^ { \circ }$ is taken as an example. As shown in Fig. 13, the variations in the horizontal and vertical PGA amplification factors with the horizontal coordinates of the slope surface under different bedrock‒overburden impedance ratios are plotted. The subfigures are grouped by various incident angles of seismic waves. The incident angle of seismic waves has a limited effect on the PGA amplification, regardless of whether the seismic wave is incident vertically or obliquely. The peak values of both the horizontal and vertical PGA amplification factors increase with increasing impedance ratio $( V _ { R } / V _ { s } )$ . The peak horizontal PGA amplification factor varies from approximately 1.5–2.7, and the peak vertical PGA amplification factor varies from approximately

![](images/07407843147cb815cd04b69080fb6aecadeac05591be054d4bca30450064f607.jpg)  
(b)  
Fig. 8. Normalized horizontal and vertical peak surface accelerations as a function of the dimensionless frequency of the incident pulse for $( \mathbf { a } ) \ : i = 3 0 ^ { \circ }$ and $( \mathbf { b } ) i =$ 60<sup>◦</sup> .

0.7–1.5. Since the impedance ratio of the bedrock to the overlying layer controls the seismic energy trapped in the overlying layer, a relatively high impedance ratio leads to greater complexity in wavefields. More specifically, consecutive reflections and interactions of seismic energy and the continuous generation of surface waves at the sharp corner of the slope result in additional aggravation of surface motion.

In the vertical direction, the higher impedance ratio $( V _ { R } / V _ { s } )$ alters the amplification pattern of the PGA from de-amplification (amplification factor less than 1.0) to actual amplification. For example, the deamplification effect is observed when the impedance ratio $V _ { R } / \ V _ { s } =$ 1.25. When the impedance ratio $V _ { R } / V _ { s } = 2 . 5 ,$ , the peak value of the vertical PGA amplification factor is close to or slightly greater than $^ { 1 . 0 , }$ , which begins to show amplification. When the impedance ratio $V _ { R } / \ V _ { s }$ $= 5 . 0 ,$ , the vertical direction shows an obvious amplification effect. The peak value of the PGA amplification factor is close to 1.5 in this case, indicating that the vertical amplification effect cannot be neglected at this stage.

![](images/8578bcf3ce25dbaa4ad05cb7894a65bb73d47edb5230bcbac6ff045eaaeaa096.jpg)  
Fig. 9. Variations in the peak values of the horizontal and vertical PGA amplification factors with the dimensionless frequency for slopes with different inclinations.

The above results show that the impedance ratio of bedrock to the overlying layer can have an important influence on both the horizontal and vertical amplification effects of the slope, whereas the effect of the incident angle of the seismic wave is of secondary importance. In particular, the vertical amplification effect of the slope is more pronounced and needs to be considered when the impedance ratio is relatively large $( \mathrm { i . e . , } V _ { R } / V _ { s } \geq 2 . 5 )$ .

To investigate the reasons for the above phenomena, snapshots of the acceleration wave fields and synthetic seismograms of the horizontal and vertical accelerations at the surface receivers are plotted in Fig. 14 for the slope configuration with $i = 4 5 ^ { \circ }$ and $h / H = 0 . 5$ . The main results of our investigation can be summarized as follows. (1) When the impedance ratio $V _ { R } / V _ { s }$ is $2 . 5 ,$ significant enhancement of the horizontal and vertical wave field motion on the slope surface is observed, and more complex Rayleigh waves and P-waves are generated. The results indicate that when the difference in wave velocity between the bedrock and overlying layer materials is large, the incident waves as well as the energy of reflected and scattered waves incoming to the upper layer of the slope continuously oscillate in the overlying layer; thus, the wave field motion at the ground surface becomes increasingly complex. (2) The material contrast between the bedrock and the overlying layer may result in enhanced vertical surface motion. Although the incident SVwave mainly causes ground motion in the horizontal direction, the vertical motion induced by the P-wave generated at the slope surface as well as the Rayleigh wave generated at the top and toe of the slope is also more significant.

## 3.4. Effect of the surface layer

This section focuses on the effects of the incident angle of the SVwave and surface layer properties on the amplification of the seismic ground motion of slopes. Two factors are considered: (1) the relative thickness $( h _ { 1 } / ( H - h ) )$ of the surface layer and (2) the impedance ratio $( V _ { s 1 } / V _ { s 2 } )$ of the surface layer to the underlying layer. The SV-Ricker wave with a dimensionless frequency of $a _ { 0 } = 2 . 0$ is used for excitation in the following simulations. The slope angle i, depth-to-bedrock ratio $h / H ,$ and bedrock-overburden impedance ratio $V _ { R } / V _ { s }$ are held constant during the numerical calculations.

Fig. 15 (a) and 15(b) show the variations in the horizontal and vertical PGA amplification factors with the horizontal coordinates of the ground surface for a slope configuration with $i = 4 5 ^ { \circ } , V _ { R } / V _ { s 2 } = 2 . 5$ , and $h / H = 0 . 5 0$ . Two relative thicknesses of the surface layer $( h _ { 1 } / ( H - h ) =$ 0.25 and 0.75) and two incident angles of the SV-wave $( \theta _ { s } = 0 ^ { \circ }$ and 15<sup>◦</sup> ) are considered. The legend of the figure represents the impedance ratio of the surface layer to the underlying layer $V _ { s 1 } / V _ { s 2 }$ , and the subplots are grouped by the incident angles $\theta _ { s } .$ . The following results can be obtained. (1) For a given relative thickness $h _ { 1 } / ( H - h )$ , the horizontal and vertical amplification effects of the soft-surface-layer slope $( V _ { s 1 } / V _ { s 2 } = 0 . 5 )$ are much stronger than those of the hard-surface-layer slope $( V _ { s 1 } / V _ { s 2 } = 2 . 0 )$ . (2) For a given incident angle $\theta _ { s } ,$ , the amplification effect of the slope becomes stronger with increasing relative thickness of the surface layer. (3) The peak values of the horizontal and vertical PGA amplification factors under obliquely incident waves are generally greater than those under vertically incident waves, i.e., obliquely incident waves significantly exacerbate the seismic ground motion of the slope surface. For example, under the vertical incidence of seismic waves, the maximum amplification factors for soft-surface-layer cases are only approximately $2 . 5 – 4 . 2$ times those for hard-surface-layer cases. Under the oblique incidence of seismic waves, the maximum amplification factors for the soft-surface-layer cases are approximately 4.8–7.4 times those for the hard-surface-layer cases. (4) The peak values of the amplification factor for slopes with a soft surface layer generally appear in the vicinity of the slope crest, whereas in the hard layer cases, the peak values of the amplification factor usually occur behind the slope toe.

Fig. 16 shows the variation in the maximum horizontal and vertical PGA amplification factors with the relative thickness of the surface layer for different impedance ratios $V _ { s 1 } / V _ { s 2 }$ . The subplots are grouped by the incident angles (θ ) of the $S V \mathrm { - } \mathbf { \vec { w a v e } }$ . The maximum horizontal and

![](images/42106fccfaf165f5dd2e35218a916cc5b12360dad4ce13f3483c0d209ed42d6e.jpg)  
Fig. 10. Snapshots of acceleration wavefields of slopes with $( \mathsf { a } ) i = 3 0 ^ { \circ }$ <sup>◦</sup> and $( \mathbf { b } ) i = 6 0 ^ { \circ }$ <sup>◦</sup> under oblique incidence of SV-waves. (c) Synthetic seismograms of horizontal and vertical accelerations of slopes with different inclinations.

![](images/7b99eff35a24ba809292836db47b07191e532e4c86d4022e00ac5d7b7cca17fe.jpg)  
(b)  
Fig. 11. Variations in the horizontal and vertical PGA amplification factors of slopes with horizontal coordinates of the slope surface for (a) $\theta _ { s } = 0 ^ { \circ }$ and $( \mathbf { b } ) \theta _ { s } = 1 5 ^ { \circ }$ .

vertical PGA amplification factors of the slope generally increase with increasing relative thickness of the surface layer. Moreover, the oblique incident wave has an enhancing effect on the horizontal and vertical amplification of the slope with a soft surface layer, but its effect on the hard-surface-layer slope is not obvious. A comparison of the maximum amplification in the horizontal direction to that in the vertical direction reveals that in the case of vertical incidence, the maximum horizontal and vertical PGA amplification factors for the soft-surface-layer slope are 2.8 and 4.3 times greater than those for the hard surface cases, respectively. Under obliquely incident seismic waves, the maximum horizontal and vertical PGA amplification factors for soft-surface-layer slopes are 4.4 and 7.4 times greater than those for hard-surface cases, respectively. In addition, for the soft-surface-layer slope under oblique incidence, extremely strong amplifications are observed not only in the horizontal direction (the maximum horizontal amplification factor reaches approximately 7.6) but also in the vertical direction (the maximum vertical amplification factor reaches approximately 5.2), which suggests that the vertical amplification for soft layer cases is dramatic and cannot be neglected.

Fig. 17 shows contour plots of the Fourier amplitude of horizontal

![](images/ed1951ee0e53305f38bb94ffc5671d235772c1977583046062f8de8e649cdbe0.jpg)  
Fig. 12. Variations in the peak values of the horizontal and vertical PGA amplification factors with respect to the dimensionless frequency for slopes with different depth-to-bedrock ratios.

![](images/878765040bb9eccc101d70739b8bfad5f40d8845bb19e9f864eba3156e765a96.jpg)  
Fig. 13. Variations in the horizontal and vertical PGA amplification factors of slopes with horizontal coordinates of the slope surface for different bedrock-overlying layer impedance ratios.

surface acceleration for slopes with soft and hard surface layers. The xaxis represents the surface receiver, and the y-axis represents the frequency. The figure shows that the peak amplitude of the ground motion on the slope surface is mainly concentrated around the dominant frequency of the incident wave (4 Hz). The Fourier amplitude of the seismic ground motion for soft-surface-layer slopes is significantly greater than that for hard-surface-layer slopes. In addition, for slopes with soft surface layers, the incident wave energy is concentrated at the upper ground surface of the slope $( \mathrm { i . e . , }$ , the area behind the slope crest), whereas for slopes with hard surface layers, the wave energy is concentrated at the lower ground surface of the slope $( \mathrm { i . e . } ,$ , the area in front of the slope toe). This variability in seismic motions on the slope surface is also important in the seismic design of slope engineering.

Snapshots of the acceleration wave field for slopes with soft and hard surface layers are shown in Fig. 18. The main results can be summarized as follows. (1) In the case of a soft-surface-layer slope $( V _ { s 1 } / \ V _ { s 2 } = 0 . 5 )$ , the energy of the incident wave is trapped within the surface layer, and the incident wave is reflected several times and interacts with the surface wave generated from the lower corner of the slope and propagates uphill. The complex scattered wave field consists mainly of Rayleigh waves that originate from the slope crest and travel along the surface.

![](images/47d078af7a8ec465d3e971075e4cb85fa385135a2bb66be243fb039f993eaa85.jpg)  
（C）  
Fig. 14. Snapshots of acceleration wavefields of slopes with (a) $V _ { R } / V _ { s } = 1$ .25 and (b) $V _ { R } / V _ { s } = 2 . 5$ under oblique incidence of SV-waves. (c) Synthetic seismograms of horizontal and vertical accelerations of slopes for different impedance ratios.

(2) In the hard-surface-layer case $( V _ { s 1 } / V _ { s 2 } = 2 . 0 )$ , the incident energy is almost completely reflected by the solid interface, and most wave energy is concentrated and oscillates in the low-velocity layer underlying the surface layer. (3) Compared with a slope with a hard surface layer, the vertical acceleration of a soft-surface-layer slope is significantly greater. This may explain the unusual vertical amplification phenomenon.

The above simulation results reveal that the stratigraphy of slopes and the incident angle of seismic waves control the acceleration amplification pattern on the slope surface and play important roles in the amplification mechanism of irregular topography. Especially for slopes containing soft surface layers, the vertical component of the acceleration can be overamplified, which should receive more attention, as evidenced by the massive destruction of building structures behind the slope crest during the 1999 Athens earthquake [9].

![](images/058e3ce8efb66ccd8fa0bc10dd455eecb39b249ab766064af22fd02a944fc877.jpg)  
(b)  
Fig. 15. Variations in the horizontal and vertical PGA amplification factors of slopes with slope surface coordinates for (a) $h _ { 1 } / ( H - h ) = 0 . 2 5$ and (b) $h _ { 1 } / ( H - h )$ $= 0 . 7 5$ .

## 4. Discussion

The topographic effect defines seismic ground motion modifications that occur as the incident seismic waves are reflected, diffracted, and

![](images/289dfa8e3d963936ddca7a77619bbac5b29ac4deadf5c600f22bab394bb04168.jpg)  
Fig. 16. Variations in the peak horizontal and vertical PGA amplification factors with the relative thickness of the surface layer for different incident angles of the SV-wave.

![](images/137d8d45d39a6cbfb1e5a94a2e41b4d87c29ee1b2ebc4f9bfd8d4e2c48e329cb.jpg)  
Fig. 17. Fourier amplitude spectra of surface horizontal acceleration for slopes with soft and hard surface layers.

scattered by the topographic surface. For slopes with various layers, incident waves undergo multiple reflections and transmissions, and most seismic energy is trapped within the lower-velocity layer, which is part of the stratigraphic effect. When seismic waves are incident at an angle, the interaction between the transmitted waves and reflected waves becomes more complicated such that further superposition of surface waves and scattering waves produces more obvious amplification on the slope surface. A schematic diagram of the propagation process of incident SV-waves through different layers of slopes is shown in Fig. 19. For SV-waves incident at an oblique angle, the critical angle of incidence should be considered, beyond which inhomogeneous waves are generated. In this study, the critical angle of the incident SV-wave is 32.3<sup>◦</sup> , while the considered incident angle of SV-wave is 15<sup>◦</sup> , which is less than the critical angle. As a result, the obtained topographic and stratigraphic amplification patterns of the slopes under the oblique incidence of seismic waves are still limited to certain conditions, and these results may not be universally applicable.

![](images/20ae4357d3f604d3aacf2cf9f047a4a8bd95300540acc43c87725f4b0df717ff.jpg)

<details>
<summary>heatmap</summary>

| Plot | Velocity Ratio (Vs1/Vs2) | Time (t) | X-Acceleration \((m/s^{2})\) Range | Y-Acceleration \((m/s^{2})\) Range |
| --- | --- | --- | --- | --- |
| (a) | 0.5 | 0.74 s | -3.22~4.35 | -2.73~3.37 |
| (a) | 0.5 | 1.13 s | -3.22~4.35 | -2.73~3.37 |
| (a) | 0.5 | 1.40 s | -3.22~4.35 | -2.73~3.37 |
| (a) | 0.5 | 1.68 s | -3.22~4.35 | -2.73~3.37 |
| (a) | 0.5 | 1.92 s | -3.22~4.35 | -2.73~3.37 |
| (a) | 0.5 | 2.40 s | -3.22~4.35 | -2.73~3.37 |
| (b) | 2.0 | 0.83 s | -3.21~4.28 | -1.65~2.56 |
| (b) | 2.0 | 0.94 s | -3.21~4.28 | -1.65~2.56 |
| (b) | 2.0 | 1.12 s | -3.21~4.28 | -1.65~2.56 |
| (b) | 2.0 | 1.30 s | -3.21~4.28 | -1.65~2.56 |
| (b) | 2.0 | 1.48 s | -3.21~4.28 | -1.65~2.56 |
| (b) | 2.0 | 1.80 s | -3.21~4.28 | -1.65~2.56 |
</details>

Fig. 18. Snapshots of acceleration wavefields of slopes with (a) soft and (b) hard surface layers under the oblique incidence of SV-waves.

![](images/4fc9d13ea3b7efa22698834837642ae816c81f75e53ab262d214e0f4330af156.jpg)

<details>
<summary>text_image</summary>

Rayleigh
wave
SV-wave
(Transmitted)
P-wave
(Transmitted)
i
Rayleigh
wave
V1
Interface
P-wave
(Reflected)
P-wave
(Incident)
SV-wave
(Reflected)
Absorbing boundary
V2
</details>

Fig. 19. Schematic diagram of the propagation process of incident SV-waves through different media of slopes.

The amplification factors are useful for seismic design in slope engineering and building engineering near slopes. They generally serve as design redundancies to ensure the safety of engineering projects. In fact, the amplification effect induced by irregular topography has been taken into account in several seismic design codes, including the 2010 Chinese seismic code (GB50011-2010) [48] and the European seismic code (EC8) [49]. The Chinese seismic code suggests an amplification factor between 1.1 and 1.6 depending on the height and inclination of the irregular topography. EC8 provides amplification factors related to topographic features for designing spectra and verifying the seismic stability of slopes. More specifically, the suggested amplification factor in EC8 is greater than or equal to 1.4 for average slope angles greater than 30<sup>◦</sup> .

Fig. 20 depicts the amplification factor range recommended by seismic codes, as well as the distribution of the maximum PGA amplification factor obtained from numerical simulations. The amplification factor of 1.1–1.6 recommended by the 2010 Chinese seismic code can cover only the partial amplification factors of numerical analysis. Especially in the horizontal direction, most of the numerical results exceed the upper limit of the range recommended by GB50011-2010, indicating that the amplification factor of 1.1–1.6 may not be sufficient when the stratigraphy and obliquely incident waves are involved. In contrast, the amplification factor prescribed by EC8 generally serves as the lower limit for the majority of horizontal amplification factors obtained from numerical modeling. Moreover, the properties of the surface layer are the factors that have the greatest influence on the amplification effect, and the soft surface layer of the slope may significantly aggravate vertical amplification. On the other hand, obliquely incident seismic waves play an important role in PGA amplification, and the maximum amplification generally appears in the case of oblique incidence. The effects of the slope angle and surface layer on the PGA amplification are more obvious than the effects of the depth-to-bedrock ratio and impedance ratio when the oblique incidence of waves is considered. Note that the comparative results are simply intended to show that the combined effects of amplification of topography and stratigraphy under obliquely incident waves may not be adequately considered in seismic codes for the seismic design of slopes. Considering that the design response spectrum is directly related to the design of structures for earthquake resistance, the spectral amplification factors of the slope should be further investigated for seismic design of buildings near the slope [50]. However, in addition to numerical modeling, trustworthy and comprehensive field observations are needed for further modification or improvement of the specific seismic code provisions.

In this study, two-dimensional (2D) slope models were employed to explore the seismic amplification of ground motion on a slope surface, which neglects the influence of model dimensionality. The threedimensional (3D) effect of topography or heterogeneities has been shown to have an impact on the amplification pattern of irregular topography, and 2D models usually underestimate the ground motion amplification in comparison with 3D models [51,52]. Therefore, the degree of ground motion amplification obtained via numerical modeling in this work may be underestimated since the three-dimensional effect was not considered. Additionally, realistic slope profiles usually consist of complex irregular surface topography. However, the simplified layered step-like slope models used to investigate the amplification effects in this study are unable to represent some complicated topographic features, thus limiting the generalization of numerical findings to a larger picture or wider applications. In conclusion, further studies involving the interaction between oblique incident seismic waves and three-dimensional slope models with realistic topography and geological conditions are necessary to examine the more reasonable and accurate seismic amplification of slopes.

## 5. Conclusion

The main objective of this study is to investigate the combined topographic and stratigraphic effects on the seismic motion of layered slopes subjected to vertically and obliquely incident SV-waves. Two numerical models of layered rock slopes are developed for dynamic analysis by considering various factors including slope angle, number of layers, properties of layer materials, and incident angle of seismic waves. The main conclusions are as follows.

1. The oblique incidence of seismic waves has a great influence on the magnitude and distribution pattern of the PGA amplification factor. Compared with vertically incident waves, obliquely incident seismic excitations can aggravate the effects of the slope angle, impedance ratio between different layers, and relative thickness of the surface layer on the amplification factor.  
2. The peak values of the horizontal and vertical PGA amplification factors on the slope surface generally increase with increasing normalized dimensionless frequency. The peak horizontal and vertical amplification factors for the layered rock slopes under vertically and obliquely incident seismic waves are in the ranges of 1.3–7.6 and 0.3–5.2, respectively. Slopes with a soft surface layer $( V _ { s 1 } / V _ { s 2 } = 0 . 5 )$ exhibit strong horizontal amplification and unusual vertical amplification, where the maximum horizontal and vertical amplification factors reach approximately 7.6 and 5.2, respectively.  
3. Among the various factors, the surface layer of the slope has the greatest influence on the amplification effect. The amplification effect generally increases with increasing thickness of the surface layer. The impedance ratio between the surface layer and the underlying layer has a significant effect on the distribution of seismic energy, and most incident energy is trapped in the low-velocity layer of the slope. When the impedance ratio $V _ { s 1 } / V _ { s 2 } = 0 . 5 ,$ , the enhanced surface motion is located behind the slope crest; when the impedance ratio $V _ { s 1 } / V _ { s 2 } = 2 . 0$ , the enhanced surface motion is located in front of the slope toe.

![](images/df6933e0bad871e71a921202afa2d6f72af9b397b66f775e16a694cfd3a22ff4.jpg)

<details>
<summary>boxplot</summary>

| Category | Effect of slope angle (Median) | Effect of depth-to-bedrock ratio (Median) | Effect of impedance ratio (Median) | Effect of surface layer (Median) |
| --- | --- | --- | --- | --- |
| Horizontal direction::1 | ~1.5 | ~1.6 | ~1.7 | ~2.8 |
| Horizontal direction::2 | ~1.4 | ~1.7 | ~2.3 | ~2.0 |
| Vertical direction::1 | ~0.6 | ~0.4 | ~1.0 | ~1.2 |
| Vertical direction::2 | ~0.6 | ~0.4 | ~1.0 | ~1.2 |
</details>

(a)

![](images/afde879d26a61f0fd85c8f6730fdf51bff98d4d35daf5d72b322bdfd2007cdc5.jpg)

<details>
<summary>boxplot</summary>

| Category | Q1 | Q2 (Median) | Q3 | IQR |
| --- | --- | --- | --- | --- |
| \(\theta s = 0{}^{\circ}\) (Grey) | ~0.8 | ~1.4 | ~1.6 | ~0.8 |
| \(\theta s = 0{}^{\circ}\) (Red) | ~0.7 | ~1.4 | ~1.9 | ~1.2 |
| \(\theta s = 0{}^{\circ}\) (Blue) | ~1.3 | ~2.0 | ~2.7 | ~1.4 |
| \(\theta s = 15{}^{\circ}\) (Grey) | ~0.9 | ~1.4 | ~1.6 | ~0.7 |
| \(\theta s = 15{}^{\circ}\) (Red) | ~0.7 | ~1.4 | ~1.9 | ~1.2 |
| \(\theta s = 15{}^{\circ}\) (Blue) | ~1.3 | ~2.0 | ~2.7 | ~1.4 |
</details>

(b)  
Fig. 20. Statistics of the maximum PGA amplification factor. The results are grouped by (a) the direction of amplification and (b) the incident angle of seismic waves.

4. The horizontal and vertical amplification effects of slopes with soft surface layers $( V _ { s 1 } / V _ { s 2 } = 0 . 5 )$ are much stronger than those of slopes with hard surface layers $( V _ { s 1 } / V _ { s 2 } = 2 . 0 )$ , and obliquely incident waves can further enhance seismic ground motions. At oblique incidence, the maximum horizontal and vertical PGA amplification factors for the soft-surface-layer slope are 4.4 and 7.4 times greater than those for the hard-surface cases, respectively. At vertical inci dence, these values are only 2.8 and 4.3, respectively.  
5. The amplification factors of the numerical results were compared with those recommended by seismic codes (GB50011-2010 and EC8). The results show that the range of the amplification factor proposed by GB50011-2010 is not sufficient to explain the amplification effect caused by the topography and stratigraphy of rock slopes under oblique incident waves, whereas the amplification factor recommended by EC8 can generally be used as the lower limit of the horizontal amplification factor of the numerical results.

## CRediT authorship contribution statement

Hui Shen: Writing – original draft, Software, Data curation. Yaqun Liu: Writing – review & editing, Funding acquisition, Conceptualization. Xinping Li: Supervision, Resources. Haibo Li: Supervision, Software. Liangjun Wang: Visualization. Wenxu Huang: Validation.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This study is funded by the National Natural Science Foundation of China under Grant Nos. 42277176 and 51679231.

## Data availability

Data will be made available on request.

## References

[1] Fan X, Juang CH, Wasowski J, Huang R, Xu Q, Scaringi G, Westen CJ, Havenith HB. What we have learned from the 2008 Wenchuan Earthquake and its aftermath: a decade of research and challenges. Eng Geol 2018;241:25–32. https://doi.org/ 10.1016/j.enggeo.2018.05.004.  
[2] Rathje EM, Stewart JP, Baturay MB, Bray JD, Bardet JP. Strong ground motions and damage patterns from the 1999 Duzce earthquake in Turkey. J Earthq Eng 2006;10:693–724. https://doi.org/10.1142/S136324690600289x.  
[3] Wang F, Fan X, Yunus AP, Siva Subramanian S, Alonso-Rodriguez A, Dai L, Xu Q, Huang R. Coseismic landslides triggered by the 2018 Hokkaido, Japan (Mw 6.6), earthquake: spatial distribution, controlling factors, and possible failure mechanism. Landslides 2019;16:1551–66. https://doi.org/10.1007/s10346-019- 01187-7.  
[4] Wasowski J, Keefer DK, Lee CT. Toward the next generation of research on earthquake-induced landslides: current issues and future challenges. Eng Geol 2011;122:1–8. https://doi.org/10.1016/j.enggeo.2011.06.001.  
[5] Assimaki D, Kausel E, Gazetas G. Wave propagation and soil–structure interaction on a cliff crest during the 1999 Athens Earthquake. Soil Dyn Earthq Eng 2005;25: 513–27. https://doi.org/10.1016/j.soildyn.2004.11.031.  
[6] Athanasopoulos GA, Pelekis PC, Leonidou EA. Effects of surface topography on seismic ground response in the Egion (Greece) 15 June 1995 earthquake. Soil Dyn Earthq Eng 1999;18:135–49. https://doi.org/10.1016/S0267-7261(98)00041-4.  
[7] Mayoral JM, De la Rosa D, Tepalcapa S. Topographic effects during the September 19, 2017 Mexico City earthquake. Soil Dyn Earthq Eng 2019;125:105732. https:// doi.org/10.1016/j.soildyn.2019.105732.  
[8] Zhang Y, Zhang J, Chen G, Zheng L, Li Y. Effects of vertical seismic force on initiation of the Daguangbao landslide induced by the 2008 Wenchuan earthquake. Soil Dyn Earthq Eng 2015;73:91–102. https://doi.org/10.1016/j. soildyn.2014.06.036.  
[9] Assimaki D, Gazetas G, Kausel E. Effects of local soil conditions on the topographic aggravation of seismic motion: parametric investigation and recorded field evidence from the 1999 Athens earthquake. Bull Seismol Soc Amer 2005;95: 1059–89. https://doi.org/10.1785/0120040055.  
[10] Sepúlveda SA, Murphy W, Petley DN. Topographic controls on coseismic rock slides during the 1999 Chi-Chi earthquake, Taiwan. Q J Eng Geol Hydrogeol 2005; 38:189–96. https://doi.org/10.1144/1470-9236/04-062.  
[11] Zhang Z, Fleurisson JA, Pellet FL. A case study of site effects on seismic ground motions at Xishan Park ridge in Zigong, Sichuan, China. Eng Geol 2018;243: 308–19. https://doi.org/10.1016/j.enggeo.2018.07.004.  
[12] Bao Y, Huang Y, Zhu C. Effects of near-fault ground motions on dynamic response of slopes based on shaking table model tests. Soil Dyn Earthq Eng 2021;149: 106869. https://doi.org/10.1016/j.soildyn.2021.106869.  
[13] Qi S, He J, Zhan Z. A single surface slope effects on seismic response based on shaking table test and numerical simulation. Eng Geol 2022;306:106762. https:// doi.org/10.1016/j.enggeo.2022.106762.  
[14] Yang G, Qi S, Wu F, Zhan Z. Seismic amplification of the anti-dip rock slope and deformation characteristics: a large-scale shaking table test. Soil Dyn Earthq Eng 2018;115:907–16. https://doi.org/10.1016/j.soildyn.2017.09.010.  
[15] Bouckovalas GD, Papadimitriou AG. Numerical evaluation of slope topography effects on seismic ground motion. Soil Dyn Earthq Eng 2005;25:547–58. https:// doi.org/10.1016/j.soildyn.2004.11.008.  
[16] Wang G, Du C, Huang D, Jin F, Koo RCH, Kwan JSH. Parametric models for 3D topographic amplification of ground motions considering subsurface soils. Soil Dyn Earthq Eng 2018;115:41–54. https://doi.org/10.1016/j.soildyn.2018.07.018.  
[17] Zhang Z, Fleurisson JA, Pellet F. The effects of slope topography on acceleration amplification and interaction between slope topography and seismic input motion. Soil Dyn Earthq Eng 2018;113:420–31. https://doi.org/10.1016/j. soildyn.2018.06.019.  
[18] Ding Y, Wang G, Yang F. Parametric investigation on the effect of near-surface soil properties on the topographic amplification of ground motions. Eng Geol 2020; 273:105687. https://doi.org/10.1016/j.enggeo.2020.105687.  
[19] Khanbabazadeh H. Nonlinearity effect on the dynamic behavior of the clayey basin edge. Geomech Eng 2024;36(4):367–80. https://doi.org/10.12989/ gae.2024.36.4.367.  
[20] Asimaki D, Mohammadi K. On the complexity of seismic waves trapped in irregular topographies. Soil Dyn Earthq Eng 2018;114:424–37. https://doi.org/10.1016/j. soildyn.2018.07.020.  
[21] Gischig VS, Eberhardt E, Moore JR, Hungr O. On the seismic response of deepseated rock slope instabilities — insights from numerical modeling. Eng Geol 2015; 193:1–18. https://doi.org/10.1016/j.enggeo.2015.04.003.  
[22] Huang D, Sun P, Jin F, Du C. Topographic amplification of ground motions incorporating uncertainty in subsurface soils with extensive geological borehole data. Soil Dyn Earthq Eng 2021;141:106441. https://doi.org/10.1016/j. soildyn.2020.106441.  
[23] Sigaki T, Kiyohara K, et al. Estimation of earthquake motion incident angle at rock site. In: Proceedings of 12th world conference on earthquake engineering. Auckland: NZ National Society for Earthquake Engineering; 2000. p. 1–8.  
[24] Huang R, Li W. Fault effect analysis of geo-hazard triggered by Wenchaun earthquake. J Eng Geol 2009;17(1):19–28 (In Chinese).  
[25] Alfaro P, Delgado J, García-Tortosa FJ, Giner JJ, Lenti L, Lopez-Casado <sup>´</sup> C, Martino S, Scarascia Mugnozza G. The role of near-field interaction between seismic waves and slope on the triggering of a rockslide at Lorca (SE Spain). Nat Hazards Earth Syst Sci 2012;12(12):3631–43. https://doi.org/10.5194/nhess-12- 3631-2012.  
[26] Vahdani S, Wikstrom S. Response of the Tarzana strong motion site during the 1994 Northridge earthquake. Soil Dyn Earthq Eng 2002;22:837–48. https://doi. org/10.1016/s0267-7261(02)00106-9.  
[27] Ashford SA, Sitar N. Analysis of topographic amplification of inclined shear waves in a steep coastal bluff. Bull Seismol Soc Amer 1997;87(3):692–700. https://doi. org/10.1785/BSSA0870030692  
[28] Fan G, Zhang LM, Li XY, Fan RL, Zhang JJ. Dynamic response of rock slopes to oblique incident SV waves. Eng Geol 2018;247:94–103. https://doi.org/10.1016/j. enggeo.2018.10.022.  
[29] Yin C, Li WH, Zhao CG, Kong XA. Impact of tensile strength and incident angles on a soil slope under earthquake SV-waves. Eng Geol 2019;260:105192. https://doi. org/10.1016/j.enggeo.2019.105192.  
[30] Shen H, Liu Y, Li H, Liu B, Xia X, Yu C. Numerical evaluation of ground motion amplification of rock slopes under obliquely incident seismic waves. Soil Dyn Earthq Eng 2024;178:108488. https://doi.org/10.1016/j.soildyn.2024.108488.  
[31] Gallipoli MR, Bianca M, Mucciarelli M, Parolai S, Picozzi M. Topographic versus stratigraphic amplification: mismatch between code provisions and observations during the L’Aquila (Italy, 2009) sequence. Bull Earthq Eng 2013;11:1325–36. https://doi.org/10.1007/s10518-013-9446-3.  
[32] Hailemikael S, Lenti L, Martino S, Paciello A, Rossi D, Mugnozza GS. Groundmotion amplification at the Colle di Roio ridge, central Italy: a combined effect of stratigraphy and topography. Geophys J Int 2016;206:1–18. https://doi.org/ 10.1093/gji/ggw120.  
[33] Meunier P, Hovius N, Haines JA. Topographic site effects and the location of earthquake induced landslides. Earth Planet Sci Lett 2008;275:221–32. https://doi. org/10.1016/j.epsl.2008.07.020.  
[34] Bourdeau C, Havenith HB. Site effects modeling applied to the slope affected by the Suusamyr earthquake (Kyrgyzstan, 1992). Eng Geol 2008;97(3–4):126–45. https:// doi.org/10.1016/j.enggeo.2007.12.009.  
[35] Rizzitano S, Cascone E, Biondi G. Coupling of topographic and stratigraphic effects on seismic response of slopes through 2D linear and equivalent linear analyses. Soil Dyn Earthq Eng 2014;67:66–84. https://doi.org/10.1016/j.soildyn.2014.09.003.  
[36] Song J, Gao Y, Feng T. Influence of interactions between topographic and soil layer amplification on seismic response of sliding mass and slope displacement. Soil Dyn Earthq Eng 2020;129:105901. https://doi.org/10.1016/j.soildyn.2019.105901.  
[37] Lee SJ, Komatitsch D, Huang BS, Tromp J. Effects of topography on seismic-wave propagation: an example from northern taiwan. Bull Seismol Soc Amer 2009;99: 314–25. https://doi.org/10.1785/0120080020.  
[38] Wang F, Ma Q, Tao D, Xie Q. A numerical study of 3D topographic site effects considering wavefield incident direction and geomorphometric parameters. Front Earth Sci 2023;10:996389. https://doi.org/10.3389/feart.2022.996389.  
[39] Komatitsch D, Tromp J. Introduction to the spectral element method for threedimensional seismic wave propagation. Geophys J Int 1999;139:806–22. https:// doi.org/10.1046/j.1365-246x.1999.00967.x.  
[40] Komatitsch D, Vilotte JP. The spectral element method: an efficient tool to simulate the seismic response of 2D and 3D geological structures. Bull Seismol Soc Amer 1998;88:368–92. https://doi.org/10.1785/bssa0880020368.  
[41] Komatitsch D, Tromp J. Spectral-element simulations of global seismic wave propagation—II. Three-dimensional models, oceans, rotation and self-gravitation. Geophys J Int 2002;150:303–18. https://doi.org/10.1046/j.1365- 246X.2002.01716.x.  
[42] Kuhlemeyer RL, Lysmer J. Finite element method accuracy for wave propagation problems. J Soil Mech Found Div 1973;99(5):421–7. https://doi.org/10.1061/ JSFEAQ.0001885.  
[43] Huang J, Zhao X, Zhao M, Du X, Wang Y, Zhang C, Zhang C. Effect of peak ground parameters on the nonlinear seismic response of long lined tunnels. Tunn Undergr Space Technol 2020;95:103175. https://doi.org/10.1016/j.tust.2019.103175.  
[44] Mahrer KD. Numerical time step instability and Stacey’s and Clayton-Engquist’s absorbing boundary conditions. Bull Seismol Soc Amer 1990;80(1):213–7. https:// doi.org/10.1785/BSSA0800010213.  
[45] Shen H, Liu Y, Li H, Liu B. Topographic effects on the seismic response of trapezoidal canyons subjected to obliquely incident SV waves. Shock Vib 2023; 2023(1):3384829. https://doi.org/10.1155/2023/3384829.  
[46] Ba Z, Zhao J, Zhu Z, Zhou G. 3D physics-based ground motion simulation and topography effects of the 05 September 2022 M<sub>W</sub> 6.6 Luding earthquake, China. Soil Dyn Earthq Eng 2023;172:108048. https://doi.org/10.1016/j. soildyn.2023.108048.  
[47] Ashford SA, Sitar N, Lysmer J, Deng N. Topographic effects on the seismic response of steep slopes. Bull Seismol Soc Amer 1997;87(3):701–9. https://doi.org/ 10.1785/BSSA0870030701.  
[48] Ministry of Construction of P. R. China. Code for seismic design of buildings, GB50011–2010. Beijing, China: China Architecture & Building Press; 2010 (in Chinese).  
[49] Eurocode 8. Design provisions for earthquake resistance of structures — Part 5: foundations, retaining structures and geotechnical aspects, ENV 1998–5. Brussels: CEN European Committee for Standardization; 2003.  
[50] Khanbabazadeh H, Iyisan R, Ozaslan B. Seismic behavior of the shallow clayey basins subjected to obliquely incident wave. Geomech Eng 2022;31(2):183–95. https://doi.org/10.12989/gae.2022.31.2.183.  
[51] Poursartip B, Kallivokas LF. Model dimensionality effects on the amplification of seismic waves. Soil Dyn Earthq Eng 2018;113:572–92. https://doi.org/10.1016/j. soildyn.2018.06.012.  
[52] Primofiore I, Baron J, Klin P, Laurenzano G, Muraro C, Capotorti F, Amanti M, Vessia G. 3D numerical modeling for interpreting topographic effects in rocky hills for Seismic Microzonation: The case study of Arquata del Tronto hamlet. Eng Geol 2020;279:105868. https://doi.org/10.1016/j.enggeo.2020.105868.