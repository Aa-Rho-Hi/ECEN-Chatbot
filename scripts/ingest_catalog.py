#!/usr/bin/env python3
"""
ingest_catalog.py — Parse and ingest TAMU ECEN catalog course descriptions.

Fetches JS-rendered catalog pages via requests-html (or uses embedded text),
parses individual ECEN courses, and upserts them into pgvector.

Run from the chatbot root:
    cd ~/Documents/Claude/Projects/chatbot
    python scripts/ingest_catalog.py
"""

import hashlib
import logging
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "crawler"))

from dotenv import load_dotenv
load_dotenv(override=True)

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PG_DSN       = os.getenv("PG_DSN", "postgresql://postgres:postgres@localhost:5433/ecen")
EMBED_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE   = 32

# ── Raw catalog text (fetched via Claude-in-Chrome from JS-rendered pages) ──────

UG_CATALOG_TEXT = """ECEN 209 Introduction to Computer Programming and Algorithms

Credits 3. 3 Lecture Hours. 1 Lab Hour. Introduction to C language programming and common algorithms; computer systems; simple C programs; basic language constructs; file I/O; modular programming and functions; arrays and matrices; pointers and strings; simple data structures; searching, sorting, and numerical algorithms; algorithmic complexity. Prerequisites: Grade of C or better in ENGR 102.

ECEN 210 Computer Programming and Algorithms

Credits 4. 3 Lecture Hours. 3 Lab Hours. Introduction to C language and common algorithms; computer systems; simple C programs; basic language constructs; file I/O; modular programming and functions; arrays and matrices; pointers and strings; simple data structures; searching, sorting, and numerical algorithms; algorithmic complexity. Prerequisite: Sophomore classification in an engineering major; Qatar campus.

ECEN 214 Electrical Circuit Theory

Credits 4. 3 Lecture Hours. 3 Lab Hours. Resistive circuits including circuit laws, network reduction, nodal analysis, mesh analysis; introduction to operational amplifiers; circuit analysis with inductors and capacitors; sinusoidal steady state circuit analysis; AC power calculations; magnetically coupled circuits; the ideal transformer; resonance; introduction to circuit simulation. Prerequisites: Grade of C or better in PHYS 207; grade of C or better in PHYS 217/ENGR 217 or ENGR 217/PHYS 217; grade of C or better in CHEM 107, CHEM 102, or CHEM 120; grade of C or better in MATH 308, or concurrent enrollment.

ECEN 215 Principles of Electrical Engineering

Credits 3. 2 Lecture Hours. 2 Lab Hours. Fundamentals of electric circuit analysis and introduction to electronics for engineering majors other than electrical and computer engineering. Prerequisites: Grade of C or better in MATH 251 or MATH 253; Grade of C or better in PHYS 207 or PHYS 208.

ECEN 222/CSCE 222 Discrete Structures for Computing

Credits 3. 3 Lecture Hours. Mathematical foundations from discrete mathematics for analyzing computer algorithms, for both correctness and performance; introduction to models of computation, including finite state machines and Turing machines. Prerequisite: Grade of C or better in MATH 142, MATH 147, MATH 151, or MATH 171.

ECEN 248 Introduction to Digital Systems Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Combinational and sequential digital system design techniques; design of practical digital systems. Prerequisite: Grade of C or better in MATH 152; grade of C or better in PHYS 207 or PHYS 208, or concurrent enrollment.

ECEN 250 Machine Learning for Electrical Engineering

Credits 3. 2 Lecture Hours. 3 Lab Hours. Engineering application-focused introduction to machine learning covering key machine learning concepts, guidance on selecting machine learning models, and application of python-based tools for data preparation, model development, and performance evaluation; practical engineering use-cases for machine learning from electronics, energy, motors, robotics, security, computer systems, and health. Prerequisites: Grade of C or better in ENGR 102; grade of C or better in MATH 251 or MATH 253.

ECEN 303 Random Signals and Systems

Credits 3. 3 Lecture Hours. Concepts of probability and random variables necessary for study of signals and systems involving uncertainty; applications to elementary problems in detection, signal processing and communication. Prerequisites: Grade of C or better in MATH 251 or MATH 253; Grade of C or better in ECEN 248.

ECEN 314 Signals and Systems

Credits 3. 3 Lecture Hours. Introduction to the continuous-time and discrete-time signals and systems; time domain characterization of linear time-invariant systems; Fourier analysis; filtering; sampling; modulation techniques for communication systems. Prerequisites: Grade of C or better in ECEN 214 or ECEN 215; grade of C or better in MATH 308; junior or senior classification.

ECEN 322 Electric and Magnetic Fields

Credits 3. 3 Lecture Hours. Vector analysis, Maxwell's equations, wave propagation in unbounded regions, reflection and refraction of waves, transmission line theory; introduction to waveguides and antennas. Prerequisites: Grade of C or better in ECEN 214, PHYS 207 or PHYS 208, and MATH 311; junior or senior classification.

ECEN 325 Electronics

Credits 4. 3 Lecture Hours. 3 Lab Hours. Electronic systems; linear circuits; operational amplifiers and applications; diodes, field effect transistors, bipolar transistors; amplifiers and nonlinear circuits. Prerequisite: Grade of C or better in MATH 311; grade of C or better in ECEN 314, or concurrent enrollment.

ECEN 326 Electronic Circuits

Credits 4. 3 Lecture Hours. 3 Lab Hours. Basic circuits used in electronic systems; differential and multistage amplifiers; output stages and power amplifiers; frequency response, feedback circuits, stability and oscillators, analog integrated circuits, active filters. Prerequisites: Grade of C or better in ECEN 314 and ECEN 325; junior or senior classification.

ECEN 333 At the Interface of Engineering and Life Sciences

Credits 3. 3 Lecture Hours. Broad overview of electrical and computer engineering principles applied to various areas of life sciences; medical imaging and biomedical signal processing; micro/nano devices and systems; computational biology and genomic signal processing. Prerequisites: Grade of C or better in ECEN 214; junior or senior classification.

ECEN 338 Electromechanical Energy Conversion

Credits 4. 3 Lecture Hours. 3 Lab Hours. Introduction to magnetic circuits, transformers, electromechanical energy conversion devices such as dc, induction and synchronous motors; equivalent circuits, performance characteristics and power electronic control. Prerequisite: ECEN 214.

ECEN 340 Electric Energy Conversion

Credits 3. 3 Lecture Hours. Fundamental topics in power and energy systems; phasors; three-phase circuits; self and mutual inductance; transformers; electromechanical systems; synchronous and induction machines; advanced concepts in electric energy conversion; DC-DC converters; inverters and rectifiers; solar and wind energy systems. Prerequisites: Grade of C or better in ECEN 214; junior or senior classification.

ECEN 350/CSCE 350 Computer Architecture and Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Computer architecture and design; use of register transfer languages and simulation tools to describe and simulate computer operation; central processing unit organization, input/output and memory system architectures. Prerequisites: Grade of C or better in ECEN 248 and CSCE 120; junior or senior classification.

ECEN 360 Computational Data Science

Credits 3. 3 Lecture Hours. Computational practice of data science through a sequence of interactive modules providing an integrated hands-on approach to its methods, tools, applications and supporting technologies including high performance and cloud computing platforms. Prerequisites: Junior or senior classification.

ECEN 370 Electronic Properties of Materials

Credits 3. 3 Lecture Hours. Introduction to basic physical properties of solid materials; engineering applications based on semiconducting, magnetic, dielectric and superconducting phenomena. Prerequisite: Grade of C or better in PHYS 222 or PHYS 309; junior or senior classification.

ECEN 403 Electrical Design Laboratory I

Credits 3. 2 Lecture Hours. 3 Lab Hours. Application of design process and project engineering as practiced in industry; team approach to the design process; development of a project proposal. Prerequisites: Grade of C or better in ECEN 314, ECEN 325, and ECEN 350/CSCE 350; senior classification.

ECEN 404 Electrical Design Laboratory II

Credits 3. 2 Lecture Hours. 3 Lab Hours. Continuation of ECEN 403; completion of project based on proposal from ECEN 403; includes testing, evaluation and report writing. Prerequisites: Grade of C or better in ECEN 403; senior classification.

ECEN 410 Medical Imaging

Credits 4. 3 Lecture Hours. 2 Lab Hours. Fundamentals of physics and the engineering principles of medical imaging systems; focus on magnetic resonance imaging, x-ray computer tomography, ultrasonography, optical imaging and nuclear medicine. Prerequisites: Grade of C or better in MATH 251 or MATH 253; ECEN 444 or grade of C or better in ECEN 314; junior or senior classification.

ECEN 419 Genomic Signal Processing

Credits 3. 3 Lecture Hours. Fundamentals of molecular biology; application of engineering principles to systems biology; topics include unearthing intergene relationships, carrying out gene-based classification of disease, modeling genetic regulatory networks. Prerequisites: Grade of C or better in ECEN 314; junior or senior classification.

ECEN 420 Linear Control Systems

Credits 3. 3 Lecture Hours. Application of state variable and frequency domain techniques to modeling, analysis and synthesis of single input, single output linear control systems. Prerequisites: Grade of C or better in ECEN 314 and MATH 308; junior or senior classification.

ECEN 423 Computer and Wireless Networks

Credits 3. 3 Lecture Hours. Fundamentals of wired and wireless computer networks; digital-data representations/transmissions, error control, MAC protocols, routing, TCP/UDP/IP, wireless TCP, queuing-delay/loss modeling, IEEE 802.11. Prerequisite: Grade of C or better in MATH 311; junior or senior classification.

ECEN 424 Fundamentals of Networking

Credits 3. 3 Lecture Hours. 1 Lab Hour. Foundations of computer networking; layered architecture of the Internet, analysis of protocols, new-age networks such as the Web and social networks. Prerequisites: Grade of C or better in ECEN 303 or STAT 211; junior or senior classification.

ECEN 425 Radio Frequency and Microwave Engineering

Credits 3. 3 Lecture Hours. Fundamental Radio Frequency (RF) and microwave circuit analysis; transmission lines, lumped elements, impedance matching; theory, analysis and design of basic RF and microwave passive circuits. Prerequisites: Grade of C or better in ECEN 322; junior or senior classification.

ECEN 427 Machine Learning

Credits 3. 3 Lecture Hours. Theoretical foundations of machine learning, pattern recognition and generating predictive models and classifiers from data; includes methods for supervised and unsupervised learning, optimization procedures and statistical inference. Prerequisites: Grade of C or better in MATH 304, MATH 311, or MATH 323; Grade of C or better in STAT 211, ECEN 303, and CSCE 121 or CSCE 120.

ECEN 428 Field Programmable Gate Arrays Information Processing Systems

Credits 4. 3 Lecture Hours. 2 Lab Hours. Signal processing and neural network implementations on field programmable gate arrays (FPGA). Prerequisites: Grade of C or better in ECEN 248 and ECEN 314; junior or senior classification.

ECEN 429 Machine Learning for Signal Processing

Credits 3. 3 Lecture Hours. Principles of pattern recognition and machine learning; applications in signal estimation, detection and classification, biomedical engineering, and cyber security of power systems. Prerequisites: Grade of C or better in ECEN 314; grade of C or better in ECEN 303 or STAT 211; junior or senior classification.

ECEN 430 Automotive Electronics

Credits 4. 3 Lecture Hours. 3 Lab Hours. Vehicular electronic systems, electric powertrain, motors and motor control, battery management, driver assistance systems (ADAS), LIDAR/Radar/Camera modules, autonomous and driverless vehicles. Prerequisites: Grade of C or better in ECEN 325; junior or senior classification.

ECEN 431 RF Circuits for Wireless Communications

Credits 4. 3 Lecture Hours. 3 Lab Hours. RF circuits for wireless communications; RF/IF amplifiers, mixers, oscillators, demodulators, and baseband amplifiers. Prerequisites: Grade of C or better in ECEN 325; senior classification.

ECEN 432 Data Conversion Systems and Circuits

Credits 4. 3 Lecture Hours. 3 Lab Hours. Sampling theory, quantization architectures, analog-to-digital and digital-to-analog converters, noise modeling, performance metrics. Prerequisites: Grade of C or better in ECEN 314 and ECEN 325; junior or senior classification.

ECEN 434 Optimization for Electrical and Computer Engineering Applications

Credits 3. 3 Lecture Hours. Principles of optimization including linear and nonlinear optimization; applications in signal estimation, routing in communication networks, economic dispatch in power systems. Prerequisites: Grade of C or better in MATH 304 or MATH 311; junior or senior classification.

ECEN 435 Audio Engineering

Credits 3. 3 Lecture Hours. Sound propagation, psychoacoustics, electroacoustics and architectural acoustics for the design and application of audio technology. Prerequisites: Grade of C or better in ECEN 314 and ECEN 325; junior or senior classification.

ECEN 438 Power Electronics

Credits 4. 3 Lecture Hours. 3 Lab Hours. Electric power conditioning and control; characteristics of solid state power switches; analysis and experiments with AC power controllers, controlled rectifiers, DC choppers and DC-AC converters. Prerequisites: Grade of C or better in ECEN 340; junior or senior classification.

ECEN 441 Electronic Motor Drives

Credits 4. 3 Lecture Hours. 3 Lab Hours. Application of semiconductor switching power converters to adjustable speed DC and AC motor drives; steady state theory and analysis of electric motion control in industrial, robotic and traction systems. Prerequisite: Grade of C or better in ECEN 340; junior or senior classification.

ECEN 444 Digital Signal Processing

Credits 4. 3 Lecture Hours. 3 Lab Hours. Digital signal processing; discrete-time signals and systems, linear shift-invariant systems, the discrete Fourier transform and fast Fourier transform algorithm, and design of digital filters. Prerequisites: Grade of C or better in ECEN 314; junior or senior classification.

ECEN 445 Applied Electromagnetic Theory

Credits 3. 3 Lecture Hours. Guided wave and wireless methods; applications of Maxwell's equations and electromagnetic wave phenomena to radiation, antennas and microwave circuit design. Prerequisites: Grade of C or better in ECEN 322; junior or senior classification.

ECEN 447 Digital Image Processing

Credits 4. 3 Lecture Hours. 3 Lab Hours. Improvement of pictorial information using spatial and frequency domain techniques; two-dimensional discrete Fourier transform; image filtering, enhancement, restoration, compression. Prerequisites: Grade of C or better in ECEN 314; junior or senior classification.

ECEN 449 Microprocessor Systems Design

Credits 3. 2 Lecture Hours. 2 Lab Hours. Introduction to microprocessors; 16/32 bit single board computer hardware and software designs; assembly language programming, stack models, subroutines and I/O processing. Prerequisites: Grade of C or better in ECEN 248; junior or senior classification.

ECEN 451 Antenna Engineering

Credits 3. 3 Lecture Hours. Antenna theory and design; theory and design of wire antennas, arrays and frequency independent antennas; computer methods for antenna design. Prerequisite: Grade of C or better in ECEN 322; junior or senior classification.

ECEN 453 Microwave Solid-State Circuits and Systems

Credits 4. 3 Lecture Hours. 3 Lab Hours. Microwave solid-state devices and circuits; theory and design of various types of active circuits; applications in radar, communication, and surveillance systems. Prerequisites: Grade of C or better in ECEN 322; junior or senior classification.

ECEN 454 Digital Integrated Circuit Design

Credits 3. 2 Lecture Hours. 2 Lab Hours. Analysis and design of digital devices and integrated circuits using MOS and bipolar technologies and computer aided simulation. Prerequisites: Grade of C or better in ECEN 214 and ECEN 248; junior or senior classification.

ECEN 455 Digital Communications

Credits 4. 3 Lecture Hours. 3 Lab Hours. Digital transmission of information through stochastic channels; signal detection, the matched-filter receiver; baseband and passband modulation, PAM, QAM, PSK, FSK; block coding, convolutional coding; synchronization. Prerequisites: Grade of C or better in ECEN 314 and ECEN 303 or STAT 211; junior or senior classification.

ECEN 459 Power System Fault Analysis and Protection

Credits 4. 3 Lecture Hours. 2 Lab Hours. General considerations in transmission and distribution of electrical energy; symmetrical components and application to analysis of power systems during fault conditions. Prerequisites: Grade of C or better in ECEN 340; junior or senior classification.

ECEN 460 Power System Operation and Control

Credits 4. 3 Lecture Hours. 2 Lab Hours. Load flow studies; power system transient stability studies; economic system loading and automatic load flow control. Prerequisites: Grade of C or better in ECEN 340; junior or senior classification.

ECEN 462 Optical Communication Systems

Credits 3. 3 Lecture Hours. Principles of optical communication systems; characteristics of optical fibers, lasers and photodetectors. Prerequisites: Grade of C or better in ECEN 322 and ECEN 370; junior or senior classification.

ECEN 464 Optical Engineering

Credits 3. 3 Lecture Hours. Ray optics; wave optics; propagation, reflection, refraction and diffraction of light; passive optical components, polarization, optical modulators, interferometers and lasers. Prerequisites: Grade of C or better in ECEN 322 and ECEN 370; junior or senior classification.

ECEN 468 Advanced Digital System Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Design, modeling and verification of complex digital systems using hardware description language and electronic system level language. Prerequisite: Grade of C or better in ECEN 248; junior or senior classification.

ECEN 469/CSCE 469 Advanced Computer Architecture

Credits 3. 3 Lecture Hours. Advanced computer architectures including memory designs, pipeline techniques and parallel structures such as out-of-order cores and multiprocessors. Prerequisite: Grade of C or better in ECEN 350/CSCE 350; junior or senior classification.

ECEN 470 Laser Principles and Applications

Credits 3. 3 Lecture Hours. Working understanding of the basic principles of laser science, the major components of laser system and their function; examples of laser applications to science, engineering, medicine and industry. Prerequisites: Grade of C or better in ECEN 322 and ECEN 370; junior or senior classification.

ECEN 471 Power Management Circuits and Systems

Credits 4. 3 Lecture Hours. 3 Lab Hours. Overview of modern semiconductor power devices, DC-DC linear regulators, switching regulators and battery chargers; analysis and design of power electronic circuits. Prerequisites: Grade of C or better in ECEN 325; junior or senior classification.

ECEN 472 Microelectronic Circuit Fabrication

Credits 4. 3 Lecture Hours. 3 Lab Hours. Fundamentals of MOS and bipolar microelectronic circuit fabrication; theory and practice of diffusion, oxidation, ion implantation, photolithography, etch. Prerequisites: Grade of C or better in ECEN 325 and ECEN 370; junior or senior classification.

ECEN 473 Microelectronic Device Design

Credits 3. 3 Lecture Hours. General processes for the fabrication of microelectronic devices; analysis of p-n junctions, bipolar transistors, and MOS capacitors and transistors. Prerequisites: Grade of C or better in ECEN 325 and ECEN 370; junior or senior classification.

ECEN 474 VLSI Circuit Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Analysis and design of monolithic analog and digital integrated circuits using NMOS, CMOS and bipolar technologies; device modeling; CAD tools; design methodologies for LSI and VLSI scale circuits. Prerequisite: ECEN 326.

ECEN 475 Introduction to VLSI Systems Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Introduction to design and fabrication of microelectronic circuits; emphasis on very large scale integration (VLSI) digital systems; use of state-of-the art design methodologies and tools. Prerequisites: Grade of C or better in ECEN 454; junior or senior classification.

ECEN 478 Wireless Communications

Credits 3. 3 Lecture Hours. Overview of wireless applications, models for wireless communication channels, modulation formats for wireless communications, multiple access techniques, wireless standards. Prerequisites: Grade of C or better in ECEN 314; junior or senior classification.

ECEN 480 RF and Microwave Wireless Systems

Credits 3. 3 Lecture Hours. Introduction to various RF and microwave system parameters, architectures and applications; theory, implementation, and design of RF and microwave systems for communications, radar, sensor, navigation, medical and optical applications. Prerequisite: Grade of C or better in ECEN 322; junior or senior classification.
"""

GRAD_CATALOG_TEXT = """ECEN 601 Mathematical Methods in Signal Processing

Credits 3. 3 Lecture Hours. Representations and algorithms for signal processing; linear algebra, vector spaces, normed and inner product spaces; projection, orthogonalization, rank-nullity theorem; matrix representations, singular value decomposition; sampling, Fourier analysis, spectral methods; statistical signal processing and linear estimation.

ECEN 602 Computer Communication and Networking

Credits 3. 3 Lecture Hours. Computer communication and computer networks; ISO seven-layer Open Systems Interconnection model; operational networks at each layer. Prerequisite: ECEN 646 or equivalent probability background.

ECEN 604 Channel Coding for Communications Systems

Credits 3. 3 Lecture Hours. Channel coding for error control, finite field algebra, block codes, cyclic codes; BCH codes; convolutional codes; Trellis coded modulation. Prerequisites: Approval of instructor and graduate classification.

ECEN 605 Linear Multivariable Systems

Credits 3. 3 Lecture Hours. Single input single output systems, multivariable systems, linear servomechanism problem and linear quadratic optimal control; classical linear control theory and modern state space control theory. Prerequisite: Graduate classification.

ECEN 606 Nonlinear Control Systems

Credits 3. 3 Lecture Hours. Techniques to analyze and synthesize nonlinear and discontinuous control systems; modern stability theory, Lyapunov Theory, adaptive control, identification and design principles. Prerequisite: ECEN 605.

ECEN 608 Modern Control

Credits 3. 3 Lecture Hours. Vector Norms; Induced Operator Norms; Lp stability; the small gain theorem; performance/robustness trade-offs; L1 and H-infinity optimal control as operator norm minimization; H2 optimal control. Prerequisite: ECEN 605 or equivalent.

ECEN 609 Adaptive Control

Credits 3. 3 Lecture Hours. Basic principles of parameter identification and parameter adaptive control; robustness and examples of instability; development of a unified approach to the design of robust adaptive schemes. Prerequisite: ECEN 605 or equivalent.

ECEN 611 General Theory of Electromechanical Motion Devices

Credits 3. 3 Lecture Hours. Winding function theory; inductances of an ideal doubly cylindrical machine; inductances of salient-pole machines; reference frame and transformation theory; dynamic equations of electric machines. Prerequisite: Approval of instructor or graduate classification.

ECEN 613 Rectifier and Inverter Circuits

Credits 3. 3 Lecture Hours. Analysis/design of single phase, three phase rectifiers; phase control and PWM rectifiers; line harmonics; power factor; harmonic standards; inverters; PWM methods; multilevel inverter. Prerequisite: ECEN 438 or approval of instructor.

ECEN 614 Power System State Estimation

Credits 3. 3 Lecture Hours. The large electric power system state estimation problem; issues of network observability; bad measurements detection/identification; sparse matrix vector techniques. Prerequisite: ECEN 460.

ECEN 615 Methods of Electric Power Systems Analysis

Credits 3. 3 Lecture Hours. Digital computer methods for solution of the load flow problem; load flow approximations; equivalents; optimal load flow. Prerequisite: ECEN 460 or approval of instructor.

ECEN 619 Internet Protocols and Modeling

Credits 3. 3 Lecture Hours. Wide spectrum of Internet protocols; analytical capabilities to evaluate the performance of complex Internet protocols; core components including TCP, UDP, IP, RIP, OSPF, BGP-4. Prerequisite: Approval of instructor.

ECEN 620 Network Theory

Credits 3. 3 Lecture Hours. Development and application of advanced topics in circuit analysis and synthesis in both the continuous and discrete time and frequency domains. Prerequisite: ECEN 326 or equivalent.

ECEN 621 Mobile Wireless Networks

Credits 3. 3 Lecture Hours. Foundations of advanced mobile wireless networks; TCP/IP over wireless links, fading-channel modeling, CDMA, OFDM, MIMO, error control, IEEE 802.11 protocols, cross-layer optimization. Prerequisites: Basic-level computer networks class or approval of instructor.

ECEN 625 Millimeter-wave Integrated Circuits

Credits 3. 3 Lecture Hours. Applications of millimeter-wave integrated circuits for wireless transceiver; principles of operation, modeling, design and fabrication of millimeter-wave CMOS, SiGe and RF MEMS circuits. Prerequisite: Graduate classification; approval of instructor.

ECEN 628 Robust and Optimal Control

Credits 3. 3 Lecture Hours. Modern design of PID controllers, robust control under parametric uncertainty and optimal control using quadratic optimization. Prerequisite: ECEN 605; graduate classification.

ECEN 629 Applied Convex Optimization

Credits 3. 3 Lecture Hours. Introduction to convex optimization including convex sets, convex functions, KKT conditions and duality, unconstrained optimization, and interior-point methods; applications in information science, digital systems, networks and learning. Prerequisites: ECEN 601 or equivalent.

ECEN 630 Analysis of Power Electronic Systems

Credits 3. 3 Lecture Hours. Analysis and control of semiconductor switching power converters using Fourier series, state-space averaging, sliding mode, and other methods. Prerequisite: Approval of instructor.

ECEN 632 Motor Drive Dynamics

Credits 3. 3 Lecture Hours. Dynamic of electric machinery; scalar control and vector control of electric machines; direct and indirect vector control for synchronous and induction motors. Prerequisites: Approval of instructor.

ECEN 635 Electromagnetic Theory

Credits 3. 3 Lecture Hours. Maxwell's equations, boundary conditions, Poynting's theorem, electromagnetic potentials, Green's functions; applications to problems involving transmission, scattering and diffraction of electromagnetic waves. Prerequisites: ECEN 322 or equivalent.

ECEN 636 Phased Arrays

Credits 3. 3 Lecture Hours. Theory and application of phased array antennas, radiators and sensors; spatial and spectral domain analysis of phased arrays; applications in radar, imaging and biomedical treatment and diagnosis. Prerequisite: ECEN 322 or equivalent.

ECEN 637 Numerical Methods in Electromagnetics

Credits 3. 3 Lecture Hours. Numerical methods of engineering electromagnetics, including finite differencing, finite difference time domain, finite elements, the method of moments and parabolic equation. Prerequisite: ECEN 322.

ECEN 638 Antennas and Propagation

Credits 3. 3 Lecture Hours. Application of Maxwell's equations to determine electromagnetic fields of antennas; radiation, directional arrays, impedance characteristics, aperture antennas. Prerequisite: ECEN 322.

ECEN 641 Microwave Solid-State Integrated Circuits

Credits 3. 3 Lecture Hours. Microwave two-terminal and three-terminal solid-state devices; theory and design of microwave mixers, detectors, modulators, switches, phase shifters, oscillators and amplifiers. Prerequisite: ECEN 322.

ECEN 642 Digital Image Processing and Computer Vision

Credits 3. 3 Lecture Hours. Digital image processing and computer vision techniques; filtering, intensity transformations, compression, restoration, segmentation, feature extraction and pattern classification. Prerequisite: ECEN 447 and ECEN 601, or approval of instructor.

ECEN 643 Electric Power System Reliability

Credits 3. 3 Lecture Hours. Design and application of mathematical models for estimating various measures of reliability in electric power systems. Prerequisite: ECEN 460 or approval of instructor.

ECEN 644 Discrete-Time Systems

Credits 3. 3 Lecture Hours. Linear discrete time systems analysis using time domain and transform approaches; digital filter design techniques with digital computer implementations. Prerequisite: ECEN 601, or approval of instructor.

ECEN 646 Probability and Random Processes for Information Science

Credits 3. 3 Lecture Hours. Concepts of probability and random processes for advanced study of information science, digital communications, networks, stochastic control and other engineering systems involving uncertainty; applications to detection, channel coding, queuing, optimization and inference.

ECEN 647 Information Theory

Credits 3. 3 Lecture Hours. Definition of information; coding of information for transmission over a noisy channel; minimum rates at which sources can be encoded; maximum rates at which information can be transmitted over noisy channels. Prerequisite: ECEN 646 or equivalent probability background.

ECEN 649 Pattern Recognition

Credits 3. 3 Lecture Hours. Optimal classification; parametric and nonparametric classification; support vector machines; neural networks; decision trees; error estimation; dimensionality reduction; model selection; Vapnik-Chervonenkis theory; Gaussian process regression. Prerequisite: Graduate classification; undergraduate probability theory and python programming skills.

ECEN 651 Microprogrammed Control of Digital Systems

Credits 4. 3 Lecture Hours. 3 Lab Hours. Hardware and software concepts in the design and construction of microprocessor-based digital systems; microprocessor architecture; bussing; interfacing; data input/output; memories; software development. Prerequisites: ECEN 350/CSCE 350 and ECEN 449 or approval of instructor.

ECEN 654 Very Large Scale Integrated Systems Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Design and fabrication of microelectronic circuits with emphasis on high-level, structured design methods for VLSI systems; design small to medium scale integrated circuits for fabrication. Prerequisites: ECEN 454 or equivalent undergraduate VLSI course.

ECEN 661 Advanced Digital Communications

Credits 3. 3 Lecture Hours. Digital communication systems; coding for discrete sources and quantization; signal spaces; modulation and demodulation, random noise; detection, error control coding, efficient decoding algorithms. Prerequisite: ECEN 646 or equivalent.

ECEN 662 Estimation and Detection Theory

Credits 3. 3 Lecture Hours. Statistical estimation and detection theory; likelihood and sufficiency principles; Bayesian point estimation; hypothesis testing; sampling methods; variational inference; application of Bayesian analysis to machine learning. Prerequisite: Graduate classification; ECEN 646.

ECEN 663 Data Compression with Applications to Speech and Video

Credits 3. 3 Lecture Hours. Characterization and representation of waveforms; digital coding of waveforms including PCM, delta modulation, DPCM, sub-band coding and transform coding; rate distortion theoretic performance bounds. Prerequisite: ECEN 601 and ECEN 646, or approval of instructor.

ECEN 665 Integrated CMOS RF Circuits and Systems

Credits 4. 3 Lecture Hours. 2 Lab Hours. Introduction to wireless communication systems at the theoretical, algorithmic and circuit levels; emphasis on simulation at the architecture, transistor levels of the communication systems; focus on circuits implementable on CMOS and BiCMOS technologies. Prerequisites: ECEN 453, ECEN 456, ECEN 474.

ECEN 666 Power System Faults and Protective Relaying

Credits 3. 3 Lecture Hours. Calculation of power system currents and voltages during faults; protective relaying principles, application and response to system faults. Prerequisite: ECEN 460 or approval of instructor.

ECEN 667 Power System Stability

Credits 3. 3 Lecture Hours. Steady-state, dynamic and transient stability of power systems; solution techniques; effect of generator control systems. Prerequisite: ECEN 460 or approval of instructor.

ECEN 669 Engineering Applications in Genomics

Credits 3. 3 Lecture Hours. Tutorial introduction to current engineering research in genomics; techniques from signal processing and control applied to intergene relationships, modeling genetic regulatory networks, and altering their dynamic behavior. Prerequisite: ECEN 605 or approval of instructor.

ECEN 671 Solid State Devices

Credits 3. 3 Lecture Hours. Development of mathematical analysis and systematic modeling of solid state devices; relationships of measurable electrical characteristics to morphology and material properties; p-n junction, bipolar and unipolar transistors. Prerequisites: Graduate classification.

ECEN 674/PHYS 674 Introduction to Quantum Computing

Credits 3. 3 Lecture Hours. Introduces the quantum mechanics, quantum gates, quantum circuits and quantum hardware of potential quantum computers; algorithms, potential uses, complexity classes, and evaluation of coherence. Prerequisites: MATH 304, PHYS 208.

ECEN 676 Advanced Computer Architecture

Credits 3. 3 Lecture Hours. Design of advanced computers for parallel processing; emphasis on overall structure; interconnection networks; shared memory and message passing architectures; multithreaded architectures; SIMD and MIMD. Prerequisite: ECEN 651 or CSCE 614 or approval of instructor.

ECEN 677 Control of Electric Power Systems

Credits 3. 3 Lecture Hours. Modeling, analysis and real-time control of electric power systems to meet the requirements of economic dispatch of voltage and power. Prerequisite: Approval of instructor.

ECEN 683 Wireless Communication Systems

Credits 3. 3 Lecture Hours. Wireless applications, modulation formats, wireless channel models and simulation techniques, digital communication over wireless channels, multiple access techniques, wireless standards. Prerequisite: ECEN 646 or approval of instructor.

ECEN 686 Electric and Hybrid Vehicles

Credits 3. 3 Lecture Hours. Fundamental concepts of electric and hybrid-electric vehicles; component requirements and system design methodologies; vehicle system analysis and simulation methods. Prerequisite: Graduate classification or approval of instructor.

ECEN 687 Introduction to VLSI Physical Design Automation

Credits 3. 3 Lecture Hours. Algorithms and techniques for VLSI design automation, including basic optimization techniques, high level synthesis, logic synthesis/verification, physical design, timing verification and optimization.

ECEN 699 Advances in VLSI Logic Synthesis

Credits 3. 3 Lecture Hours. Logic representation, manipulation, and optimization; combinational and sequential logic; Boolean function representation schemes; exact and heuristic two-level logic minimization; multi-level logic representation and minimization. Prerequisites: Approval of instructor and graduate classification.

ECEN 710 Switching Power Supplies

Credits 3. 3 Lecture Hours. Operating principles of switching power supplies; analysis and in-depth design of buck, boost, forward, flyback, half and full bridge switching regulators; transformer and magnetic design; feedback loop stabilization. Prerequisites: ECEN 438 or equivalent, approval of instructor.

ECEN 711 Sustainable Energy and Vehicle Engineering

Credits 3. 3 Lecture Hours. Forms of sustainable and unsustainable energy resources; specific problems of sustainable transportation energy; issues related to energy efficiency, life cycle analysis, global warming, pollution, economic and social considerations. Prerequisite: Graduate classification in engineering.

ECEN 712 Power Electronics for Photovoltaic Energy Systems

Credits 3. 3 Lecture Hours. Electrical characteristics of solar photovoltaic sources; requirements for grid-connection and power electronic circuits and controls needed for the interconnection and control. Prerequisite: ECEN 438 or approval of instructor.

ECEN 713 Data Sciences and Applications for Modern Power

Credits 3. 3 Lecture Hours. Introduction to the foundation of high dimensional statistics; data analytical tools necessary to model and operate a modern power system; projects offer realistic data sets to construct tools and models for smart grid operations. Prerequisite: ECEN 420 or ECEN 460, or equivalent.

ECEN 716 Smart Power Distribution Systems Analysis and Operation

Credits 3. 3 Lecture Hours. Study of element models in active power distribution systems in smart grid operation; emerging topics such as integrating solar panels, microgrid operations, advanced metering infrastructure, and restoration and outage management. Prerequisites: Graduate classification.

ECEN 719 Advanced Digital System Design

Credits 4. 3 Lecture Hours. 3 Lab Hours. Introduction to the design, modeling and verification of complex digital systems using hardware description language and electronic system level language. Prerequisites: Graduate classification; basic knowledge of digital logic and integrated circuits.

ECEN 720 High-Speed Links Circuits and Systems

Credits 4. 3 Lecture Hours. 3 Lab Hours. System and circuit design of high-speed electrical and optical link systems; channel properties, communication techniques, and circuit design of drivers, receivers, equalizers, and synchronization systems. Prerequisite: ECEN 474.

ECEN 722 Field Programmable Gate Arrays Information Processing Systems

Credits 4. 3 Lecture Hours. 2 Lab Hours. Signal processing and neural network implementations on field programmable gate arrays (FPGA); FPGA designs of digital filters, Fourier transform, JPEG decoding, fast convolution, Kalman filter and Viterbi decoding. Prerequisites: Graduate classification.

ECEN 723 Introduction to Formal Verification

Credits 3. 3 Lecture Hours. Formal verification techniques for hardware and concurrent systems; binary decision diagrams; Boolean satisfiability; equivalence checking; model checking; temporal logic; design assertions; probabilistic model checking. Prerequisites: Graduate classification.

ECEN 732 Online Decision Making and Learning

Credits 3. 3 Lecture Hours. Design and analysis of online decision making policies; topics include online learning, prima-dual techniques, online convex optimization and multi-armed bandit problem. Prerequisites: Graduate classification.

ECEN 735 Electromagnetic Field Theory

Credits 3. 3 Lecture Hours. Methods in wave propagation, diffraction and scattering analysis, including surface waves, creeping waves, surface plasmons and complex environments; applications to macroscopic and nano technology. Prerequisite: ECEN 635 or equivalent.

ECEN 738 Power Electronics

Credits 4. 3 Lecture Hours. 3 Lab Hours. Electric power conditioning and control; characteristics of solid state power switches; analysis and experiments with AC power controllers, controlled rectifiers, DC choppers and DC-AC converters. Prerequisite: Graduate classification or approval of instructor.

ECEN 740 Machine Learning Engineering

Credits 3. 3 Lecture Hours. Fundamental theory for learning supervised classification-regression models; covers Bayes classifier, neural networks, Convolutional Neural networks, Auto-encoders, Generative Adversarial Networks, support vector machines, kernel-based methods, boosting, Gaussian process-based learning, word embeddings, recurrent neural networks, decision trees. Prerequisite: ECEN 303, MATH 411, STAT 614, STAT 615, or approval of instructor.

ECEN 741 Electronic Motor Drives

Credits 4. 3 Lecture Hours. 3 Lab Hours. Application of semiconductor switching power converters to adjustable speed DC and AC motor drives; steady state theory and analysis of electric motion control in industrial, robotic and traction systems. Prerequisite: Graduate classification.

ECEN 743 Reinforcement Learning

Credits 3. 3 Lecture Hours. Introduction to the theory and practice of reinforcement learning; including Markov decision processes, dynamic programming, Q-Learning, policy gradient algorithms, neural networks, deep reinforcement learning, imitation learning and multi-agent learning. Prerequisite: Graduate classification.

ECEN 744 Scientific Machine Learning

Credits 3. 3 Lecture Hours. Introduction to the algorithmic and computational foundations of regularizing scientific laws in machine learning; ODE and PDE discretization; automatic differentiation; physics-informed neural networks; physics-informed Gaussian processes; operator learning. Prerequisite: Graduate classification or approval of instructor.

ECEN 748 Data Stream Algorithms and Applications

Credits 3. 3 Lecture Hours. Study of algorithms to sample, sketch and summarize high rate data streams; applications to measuring internet traffic and transactional graph streaming data. Prerequisites: Graduate classification; ECEN 303 or previous undergraduate or graduate course in probability or statistics.

ECEN 750 Design and Analysis of Communication Networks

Credits 3. 3 Lecture Hours. Analytical approach to understanding resource allocation on the Internet; resource allocation, congestion control protocols, stochastic approach to understanding system performance. Prerequisite: ECEN 646 or some probability background.

ECEN 752 Advances in VLSI Circuit Design

Credits 3. 3 Lecture Hours. Gate and wire delays, CMOS transistors, DC and AC characteristics, VLSI fabrication, DRAM, SRAM and FLASH memory design, leakage and dynamic power, sub-threshold computation, clocking, process variation and compensation. Prerequisites: Graduate classification or approval of instructor.

ECEN 753 Theory and Applications of Network Coding

Credits 3. 3 Lecture Hours. Fundamentals of network coding including concepts, models, linear and non-linear codes, code design; wireless network coding; network coding for storage. Prerequisite: Graduate classification or approval of instructor.

ECEN 754 Optimization for Electrical and Computer Engineering Applications

Credits 3. 3 Lecture Hours. Principles of optimization including linear and nonlinear optimization; applications in signal estimation, routing in communication networks, economic dispatch in power systems. Prerequisites: MATH 304 or MATH 311; MATH 251 or graduate classification.

ECEN 755 Stochastic Systems

Credits 3. 3 Lecture Hours. Principles of stochastic systems including performance evaluation, estimation, control, scheduling, identification and adaptation; applications in communication networks and control. Prerequisites: MATH 411; approval of instructor and graduate classification.

ECEN 756 Game Theory

Credits 3. 3 Lecture Hours. Fundamentals of game theory, strategic behavior and concepts of equilibria; algorithms and learning methods to compute such equilibria. Prerequisite: Graduate classification.

ECEN 757/CSCE 678 Distributed Systems and Cloud Computing

Credits 3. 3 Lecture Hours. Fundamental concepts of distributed systems with a focus on cloud computing; MapReduce, synchronization, peer-to-peer systems, election, distributed agreement, replication, job assignment.

ECEN 758 Data Mining and Analysis

Credits 3. 3 Lecture Hours. Broad overview of data mining, integrating related concepts from machine learning and statistics; exploratory data analysis, pattern mining, clustering and classification; applications to scientific and online data.

ECEN 759/CYBR 630 Hardware Security

Credits 3. 3 Lecture Hours. Cryptography and cryptographic algorithms such as AES, DES; techniques to optimize hardware implementation; side-channel attacks and countermeasures; supply-chain vulnerabilities; hardware Trojans; physical unclonable function. Prerequisites: ECEN 350/CSCE 350 or approval of instructor.

ECEN 760 Introduction to Probabilistic Graphical Models

Credits 3. 3 Lecture Hours. Broad overview of probabilistic graphical models, including Bayesian networks, Markov networks, conditional random fields, and factor graphs; relevant inference and learning algorithms. Prerequisites: Undergraduate level probability theory; basic programming skill.

ECEN 765 Machine Learning with Networks

Credits 3. 3 Lecture Hours. Scientific analysis of large-scale data; introduction to advanced methods designed to analyze structured data represented as networks. Prerequisite: Approval of instructor.

ECEN 767 Harnessing Solar Energy: Optics, Photovoltaics and Thermal Systems

Credits 4. 3 Lecture Hours. 3 Lab Hours. Solar radiation characteristics and measurement; optical coatings; concentrating optics; photovoltaic cells, modules and systems overview; introduction to solar thermal systems. Prerequisite: Graduate classification or approval of instructor.

ECEN 768 Bioelectronics

Credits 3. 3 Lecture Hours. Basic biological systems from individual neuron to neural networks in the brain/nervous system leveraging engineering principles; applications include biosensors including electrodes, chemical, mechanical and optical sensors.

ECEN 771 Fluctuations and Noise Electronics

Credits 3. 3 Lecture Hours. Introduction to the research of Noise and Fluctuations in electronics and other systems; applications in secure communications, microprocessors, quantum information, mesoscopic systems, chemical sensing. Prerequisite: Approval of instructor.

ECEN 772 Introduction to Microelectromechanical Devices and Systems

Credits 4. 3 Lecture Hours. 2 Lab Hours. Broad overview of MEMS (microelectromechanical systems); fundamental working principles, designs and fabrication techniques; special topics discussing latest important applications in different fields. Prerequisite: Graduate classification in engineering.

ECEN 774 Laser Principles and Applications

Credits 3. 3 Lecture Hours. Quantum properties of light and matter as related to optical and optoelectronic devices such as lasers; Maxwell's equations, classical optics and optical devices; basic quantum theory of light and atoms; laser resonators and short pulse generation.

ECEN 776 Unconditionally Secure Electronics

Credits 3. 3 Lecture Hours. Data security; cryptography; key exchange; unconditional (information-theoretic) security; quantum key distribution; the Kirchhoff-law-Johnson-noise (KLJN) key exchange; schemes, protocols, attacks, defense, privacy amplification. Prerequisites: ECEN 214, ECEN 303, or STAT 211; graduate classification.
"""


# ── Parsing ──────────────────────────────────────────────────────────────────────

def parse_courses(text: str) -> list[dict]:
    """Split raw catalog text into individual course dicts."""
    # Pattern: ECEN NNN [/CSCE NNN] Course Title (on a line by itself)
    pattern = re.compile(r'^(ECEN\s+\d+(?:/[A-Z]+\s+\d+)?)\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    courses = []
    for i, m in enumerate(matches):
        start = m.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        courses.append({
            "number": m.group(1).strip(),
            "name":   m.group(2).strip(),
            "text":   block,
        })
    return courses


# ── DB helpers ────────────────────────────────────────────────────────────────────

def get_conn():
    conn = psycopg2.connect(PG_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def embed_texts(embedder, texts: list[str]) -> list[list[float]]:
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = embedder.encode(texts[i:i + BATCH_SIZE], normalize_embeddings=True)
        vecs.extend(batch.tolist())
    return vecs


def upsert_chunks(conn, chunks: list[dict], vectors: list[list[float]]):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    with conn.cursor() as cur:
        for chunk, vec in zip(chunks, vectors):
            row_id = int(hashlib.md5(chunk["chunk_id"].encode()).hexdigest(), 16) % (2**63)
            cur.execute("""
                INSERT INTO ecen_docs
                    (id, chunk_id, url, title, section, text, content_hash, last_indexed, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    url          = EXCLUDED.url,
                    title        = EXCLUDED.title,
                    section      = EXCLUDED.section,
                    text         = EXCLUDED.text,
                    content_hash = EXCLUDED.content_hash,
                    last_indexed = EXCLUDED.last_indexed,
                    embedding    = EXCLUDED.embedding;
            """, (
                row_id, chunk["chunk_id"], chunk["url"], chunk["title"],
                chunk["section"], chunk["text"], chunk["content_hash"],
                now, vec,
            ))
    conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    log.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    conn = get_conn()

    datasets = [
        {
            "text":    UG_CATALOG_TEXT,
            "url":     "https://catalog.tamu.edu/undergraduate/engineering/electrical-computer/#coursestext",
            "section": "academics",
            "level":   "undergraduate",
        },
        {
            "text":    GRAD_CATALOG_TEXT,
            "url":     "https://catalog.tamu.edu/graduate/colleges-schools-interdisciplinary/engineering/electrical-computer/#coursestext",
            "section": "academics",
            "level":   "graduate",
        },
    ]

    total_upserted = 0
    for ds in datasets:
        courses = parse_courses(ds["text"])
        log.info("Parsed %d %s courses", len(courses), ds["level"])

        chunks = []
        for c in courses:
            chunk_id    = hashlib.md5(f"catalog_{c['number']}_{ds['level']}".encode()).hexdigest()
            content_hash = hashlib.md5(c["text"].encode()).hexdigest()
            chunks.append({
                "chunk_id":     chunk_id,
                "url":          ds["url"],
                "title":        f"{c['number']} {c['name']} | TAMU ECE Catalog",
                "section":      ds["section"],
                "text":         c["text"],
                "content_hash": content_hash,
            })

        texts  = [ch["text"] for ch in chunks]
        log.info("Embedding %d chunks...", len(texts))
        vectors = embed_texts(embedder, texts)

        upsert_chunks(conn, chunks, vectors)
        log.info("Upserted %d %s course chunks.", len(chunks), ds["level"])
        total_upserted += len(chunks)

    conn.close()
    log.info("Done. Total upserted: %d chunks.", total_upserted)


if __name__ == "__main__":
    main()
