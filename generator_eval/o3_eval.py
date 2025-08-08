from openai import AzureOpenAI
import json
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse
import time
import math
import re
from transformers import AutoTokenizer



CONCISE_ZERO_SHOT = """### Question: [HERE_IS_THE_QUESTION]

### Candidate 1: [HERE_IS_THE_GROUND_TRUTH]

### Candidate 2: [HERE_IS_THE_CANDIDATE]

### Guidelines: For the above question, please verify if candidate 1 and candidate 2 predictions are equivalent or not.
DO NOT ATTEMPT TO SOLVE the question by yourself; instead focus on checking if the two candidates are equivalent.
If the two candidates are equivalent, output \"Final Judgment: Yes <End of Judgment>\". If not, output \"Final Judgment: No <End of Judgment>\". Most importantly, DO NOT MAKE a judgment first. Instead, first reason about whether the candidates are equivalent or not based on the specified rules above (read through all of them, not only one), and then output the final judgment.

### Reasoning: 
"""

DETAILED_ZERO_SHOT = """### Question: [HERE_IS_THE_QUESTION]

### Candidate 1: [HERE_IS_THE_GROUND_TRUTH]

### Candidate 2: [HERE_IS_THE_CANDIDATE]

### Guidelines: For the above question, please verify if candidate 1 and candidate 2 predictions are equivalent or not.
DO NOT ATTEMPT TO SOLVE the question by yourself; instead focus on checking if the two candidates are equivalent based on the following rules:

<Rule 1> If the two quantities are being compared, first check if both candidates are using the same unit. If the units are different, convert them to the same unit if possible and compare the values. If one or both candidates have omitted a unit, refer to the question to determine if multiple units could reasonably be used; if so, and the omission causes ambiguity or confusion, the answers should be considered inequivalent. If the two quantities are of different types (e.g., count and length), check if a physical or mathematical relationship exists that allows conversion. If a physical relationship (such as a device construction formula) applies, use it to relate the quantities. If only a mathematical relationship (such as unit conversion) applies, use that. If neither applies, the quantities are not equivalent.
<Rule 2> There are questions that consist of multiple sub-problems or asks for multiple answers, check how many answers are required from the question before determining the equivalence of two candidates. If the question asks for multiple answers and one candidate provides only a fewer number of answer than requested while the other candidate provides all multiple answers (regardless of whether the answers are identical, repeated, or different), they are not considered equivalent. In case both candidates have provided all requested multiple answers, check if each and every one of answers from both candidates are exactly identical. If there is at least one discrepancy, they are not equivalent. Refer to the other rules when comparing each sub-answers. Importantly, note that it is allowed for a candidate to provide more answers than the number of sub-problems only if some sub-problems require multiple components or equations (for example, both x and y components, or multiple equations for a vector quantity). In such cases, all required components or parts for each sub-problem must be present in the candidate’s answer. If a candidate omits any required component (such as providing only the x-component when both x and y are required), the answers are not equivalent. When comparing, ensure that the structure and number of answers match the requirements of the question, including cases where a single sub-problem expects multiple distinct answers.
<Rule 3> When the candidates are a list of numbers, determine from the question whether order and multiplicity matter. If order matters, the lists must match exactly in both order and value. If order does not matter, compare as sets, ignoring order and duplicates. If the question expects multiplicity, then duplicates must match exactly. If the number of elements differs, the answers are not equivalent unless the question allows for variable-length answers.
<Rule 4> If candidate 1 and candidate 2 are real numbers, determine whether their results are the same by comparing both to 6 significant digits. Do not determine that the two candidates are equivalent because they have similar values but instead, AND ALWAYS EXPRESS UP TO 6 SIGNIFICANT DIGITS FOR STRICT COMPARISON (including all the 0s). If both answers match to 6 significant digits, they are considered the same; otherwise, they are different. Do not round to fewer digits, even if one answer is given with less precision—always compare at 6 significant digits. When comparing numbers in scientific notation, first express both numbers with the same exponent (matching the smaller exponent of the two) before comparing the significant digits.
<Rule 5> For multiple-choice questions, answers may be given as letters (e.g., A/B/C/D/E/F, (1)/(2)/(3)/(4)), as the option text itself, or a mix of both. Two candidates are considered equivalent only if they select the same option, either by letter or by exactly matching the option text (e.g., (A) and (C) should always be not equivalent). Ensure that the selected answer matches the corresponding option in the question—if the answer is given as text, verify that it aligns with the correct lettered choice in the problem. If the candidate’s answer consists only of text that does not match any of the provided options, it should be considered inequivalent, as the candidate’s response is not a valid choice. In case one of the candidate does not provide a prediction at all (e.g., an empty string), the two candidates should always be marked as not equivalent. Moreover, if one candidate made chose multiple options at once and the other candidate only chose one of the options, they are not equivalent.
<Rule 6> When there is a type mismatch between candidates—such as when one answer is a text span and the other is an approximate equality—these are not considered equivalent. In particular, if one candidate does not provide a conclusive answer (for example, by not explicitly stating the actual numerical value and instead offering only relevant context or description), do not consider the answers equivalent. Refer to rule 8 about \"lazy reasoning\" behavior, which should be considered when determining equivalence of two candidates.
<Rule 7> In case one of the candidate provides a whole solution or thinking process instead of the final prediction, the two candidates are not equivalent. This is to prevent the confusion of determining the equivalence of the two candidates. There could be cases where the final prediction is a text span or there are one or two sentences of explanation, which you could refer to the other rules to determine equivalence. However, if one candidate provides a detailed step-by-step solution, they are not equivalent.
<Rule 8> For questions that require writing an interval, equation, piecewise function, set, ideal, or matrix, reason carefully about whether the two candidates are mathematically equivalent, even if they look different. Consider standard mathematical conventions and context. Always check if different notations represent the same mathematical object in the context of the question. Note that in case one of the candidate omits a crucial component requested by the question, the two candidates are not equivalent. Additionally, if a candidate provides only a general, qualitative, or incomplete answer (for example, by giving a vague description or a general formula instead of the explicit, detailed result required), this is considered \"lazy reasoning\" and the answers are not equivalent, even if the general answer is related to the topic.
<Rule 9> For questions that require a text span (e.g., explaining a concept, providing examples, naming an entity), do not judge which is better; instead, focus strictly on whether both candidates contain the same essential information and reach the same level of completion. Both answers must include all information explicitly required by the question and must be equally complete (e.g., both provide a final answer, or both provide a method, as required). Extra detail is allowed, but omission of any required element means the answers are not equivalent.
<Rule 10> If the two candidates are equivalent, output \"Final Judgment: Yes <End of Judgment>\". If not, output \"Final Judgment: No <End of Judgment>\". Most importantly, DO NOT MAKE a judgment first. Instead, first reason about whether the candidates are equivalent or not based on the specified rules above (read through all of them, not only one), and then output the final judgment.

### Reasoning: 
"""

DETAILED_FEW_SHOT = """### Question: [HERE_IS_THE_QUESTION]

### Candidate 1: [HERE_IS_THE_GROUND_TRUTH]

### Candidate 2: [HERE_IS_THE_CANDIDATE]

### Guidelines: For the above question, please verify if candidate 1 and candidate 2 predictions are equivalent or not.
DO NOT ATTEMPT TO SOLVE the question by yourself; instead focus on checking if the two candidates are equivalent based on the following rules:

<Rule 1> If the two quantities are being compared, first check if both candidates are using the same unit. If the units are different, convert them to the same unit if possible and compare the values. If one or both candidates have omitted a unit, refer to the question to determine if multiple units could reasonably be used; if so, and the omission causes ambiguity or confusion, the answers should be considered inequivalent. If the two quantities are of different types (e.g., count and length), check if a physical or mathematical relationship exists that allows conversion. If a physical relationship (such as a device construction formula) applies, use it to relate the quantities. If only a mathematical relationship (such as unit conversion) applies, use that. If neither applies, the quantities are not equivalent. [Example 1-1] \"289 rulings\" (which is a count of lines on a diffraction grating) and \"3.46 × 10^{-3} m\" (which is a length) are not equivalent unless the spacing between the rulings is specified, since a physical relationship (total width = number of rulings × spacing) is required to connect them. Without this information, they should be considered inequivalent. [Example 1-2] \"0.012 m\" and \"1.2 cm,\" are equivalent because they represent the same value in different units. [Example 1-3] \"20 micro ohm\" and \"0.2\" (which has no specified unit), are inequivalent since it is ambiguous whether \"0.2\" denotes 0.2 ohm or 0.2 micro ohm.
<Rule 2> There are questions that consist of multiple sub-problems or asks for multiple answers, check how many answers are required from the question before determining the equivalence of two candidates. If the question asks for multiple answers and one candidate provides only a fewer number of answer than requested while the other candidate provides all multiple answers (regardless of whether the answers are identical, repeated, or different), they are not considered equivalent. In case both candidates have provided all requested multiple answers, check if each and every one of answers from both candidates are exactly identical. If there is at least one discrepancy, they are not equivalent. Refer to the other rules when comparing each sub-answers. Importantly, note that it is allowed for a candidate to provide more answers than the number of sub-problems only if some sub-problems require multiple components or equations (for example, both x and y components, or multiple equations for a vector quantity). In such cases, all required components or parts for each sub-problem must be present in the candidate’s answer. If a candidate omits any required component (such as providing only the x-component when both x and y are required), the answers are not equivalent. When comparing, ensure that the structure and number of answers match the requirements of the question, including cases where a single sub-problem expects multiple distinct answers. [Example 2-1] \"1. 70.5, 2. [0, 5]\" and \"70.5\" are not equivalent since the second candidate provided only one answer. [Example 2-2] \"answer 1: C \ answer 2: C\" and \"C\" are not equivalent since the first candidate provided only one answer. [Example 2-3] \"No, it is not possible to monochromatize the beam with this configuration.\" and \"- Answer 1: $\lambda = \frac{2 \sin \theta}{mN}$ - Answer 2: Not possible\", are not equivalent since the first candidate provided only one answer. [Example 2-4] \"B, C, D\" and \"answer1: B, answer2: C, answer3: C\", are not equivalent because the last answer3 differs. [Example 2-5] \"* answer 1: x * answer 2: L = \frac{1}{2} m \dot{x}^2 \left(1 + \frac{4x^2}{a^2}\right) - \frac{mgx^2}{a} * answer 3: (0, 0) * answer 4: \ddot{x} + \frac{2gx}{a} = 0 * answer 5: x = A \cos \left( \sqrt{\frac{2g}{a}} t + \epsilon \right)\" and \"The solution for the small oscillations is: $$q(t) = A \cos \left( \sqrt{\frac{2g}{a}} t \right) + B \sin \left( \sqrt{\frac{2g}{a}} t \right)$$ where $A$ and $B$ are constants determined by the initial conditions.\" are not equivalent since the former provides five answers while the latter provides only one. [Example 2-6] For the question \"A phonograph turntable in the $xy$ plane revolves at constant angular velocity $\omega$ around the origin. A small body sliding on the turntable has location $\mathbf{x}(t) = (x(t), y(t), 0)$. Here $x$ and $y$ are measured in an inertial frame, the lab frame. There are two forces in the lab frame: an elastic force of magnitude $k|\mathbf{x}|$ towards the origin, and a frictional force $-c(\dot{\mathbf{x}} - \mathbf{v})$, where $c$ is a constant and $\mathbf{v}$ is the velocity of the turntable at the body's location. (a) If the body is observed to stay at a fixed off-center point on the turntable (i.e. it is at rest with respect to the turntable), how big is $k$? (b) Assume $k$ has the value you found in (a). Solve for $\mathbf{v}(t) = \dot{\mathbf{x}}(t)$ with general initial conditions. (c) In (b), find $\mathbf{x}(t)$. Describe $\mathbf{x}(t)$ in words and/or with a rough sketch.\", candidate \"* answer 1: k = m\omega^2 * answer 2: \dot{x} = [\dot{x}_0 \cos(2\omega t) + \dot{y}_0 \sin(2\omega t)]e^{-ct/m} * answer 3: \dot{y} = [-\dot{x}_0 \sin(2\omega t) + \dot{y}_0 \cos(2\omega t)]e^{-ct/m} * answer 4: x = x_0 + \frac{m(c\dot{x}_0 + 2m\omega \dot{y_0})}{c^2 + 4m^2\omega^2} - \left[ \frac{m(\dot{x}_0 + 2m\omega \dot{y}_0)}{c^2 + 4m^2\omega^2} \cos(2\omega t) - \frac{m(2m\omega \dot{x}_0 - c\dot{y}_0)}{c^2 + 4m^2\omega^2} \sin(2\omega t) \right] e^{-ct/m} * answer 5: y = y_0 - \frac{m(2m\omega \dot{x}_0 - c\dot{y}_0)}{c^2 + 4m^2\omega^2} + \left[ \frac{m(2m\omega \dot{x}_0 - c\dot{y}_0)}{c^2 + 4m^2\omega^2} \cos(2\omega t) + \frac{m(c\dot{x}_0 + 2m\omega \dot{y}_0)}{c^2 + 4m^2\omega^2} \sin(2\omega t) \right] e^{-ct/m}\" and candidate \"(a) \( k = m \omega^2 \), (b) \[\mathbf{v}(t) = A (\alpha + \beta i) e^{(\alpha + \beta i)t} + B (\alpha - \beta i) e^{(\alpha - \beta i)t}\], (c) The body will oscillate with decreasing amplitude due to the damping effect of the frictional force and eventually settle into a steady-state position given by: \[\mathbf{x}(t) = A e^{(\alpha + \beta i)t} + B e^{(\alpha - \beta i)t} + \frac{c \mathbf{v}}{m \omega^2}\]\" are different not because the former candidate has provided five sub-answers (this is actually not problematic since answer 2,3 responds to sub-problem (b) and answer 4,5 responds to sub-problem (c)), but because the latter candidate does not provide all the required components for each sub-problem as specified by the question. The lack of explicit component-wise solutions for both velocity and position in the extracted answer is the main reason the answers are not equivalent. [Example 2-7] \"\begin{aligned} \text{answer 1:} & \text{ No} \text{answer 2:} & \text{ Yes} \end{aligned}\" and \"1. No, the existence of F does not imply the existence of G. 2. No, the left derived functor RHom_D(F(-), -) is not right adjoint to the left derived functor LHom_C(-, G(-)).\" are not equivalent, since the candidates have different predictions for the second sub-problem. For such True/False or Yes/No questions, do not make any additional subjective judgment about why the two candidates could be treated as equivalent; they are strictly different. Note that \"Yes\" and \"True\" or \"No\" and \"False\" can be treated as equivalent, but do not overthink when comparing across types (\"Yes\" and \"False\" are not equivalent, \"Yes\" and \"No\" are not equivalent, \"True\" and \"False\" are not equivalent, and \"True\" and \"No\" are also not equivalent). [Example 2-8] \"C = \text{unit square}, A = (0,0), B = (1,0), C = (0,1), O = \left( \frac{1}{3}, \frac{1}{3} \right)\" and \"A simple convex set \( C \) is a triangle with vertices at \( (0, 0) \), \( (1, 0) \), and \( (0, 1) \). The coordinates of the center of mass \( O \) are \( \left(\frac{1}{3}, \frac{1}{3}\right) \).\" are equivalent because both candidates use the same three points (0,0),(1,0),(0,1) and compute the center of mass as (\frac{1}{3}, \frac{1}{3}), differing only in whether they describe the convex set as a unit square or a triangle with those vertices.
<Rule 3> When the candidates are a list of numbers, determine from the question whether order and multiplicity matter. If order matters, the lists must match exactly in both order and value. If order does not matter, compare as sets, ignoring order and duplicates. If the question expects multiplicity, then duplicates must match exactly. If the number of elements differs, the answers are not equivalent unless the question allows for variable-length answers. [Example 3-1] If the question asks for the roots in increasing order, \"0, 3\" and \"3, 0\" are not equivalent. [Example 3-2] If the question asks to \"list all solutions,\" \"0, 3\" and \"3, 0\" are equivalent. [Example 3-3] If the question asks to calculate the value of a and b, \"a=0, b=3\" and \"0, 3\" are not equivalent because it is not clear which value corresponds to a and which corresponds to b in the latter case.
<Rule 4> If candidate 1 and candidate 2 are real numbers, determine whether their results are the same by comparing both to 6 significant digits. Do not determine that the two candidates are equivalent because they have similar values but instead, AND ALWAYS EXPRESS UP TO 6 SIGNIFICANT DIGITS FOR STRICT COMPARISON (including all the 0s). If both answers match to 6 significant digits, they are considered the same; otherwise, they are different. Do not round to fewer digits, even if one answer is given with less precision—always compare at 6 significant digits. When comparing numbers in scientific notation, first express both numbers with the same exponent (matching the smaller exponent of the two) before comparing the significant digits. Both the significant digits and the exponent must match for the answers to be considered the same. [Example 4-1] 0.15 and 3/20 are equivalent (since 3/20 = 0.150000). [Example 4-2] 0.788 and 0.7884 are not equivalent (since 0.788000 and 0.788400 are different at 6 significant digits). [Example 4-3] 3.00000×10^15 and 3.35000×10^15 are not equivalent (since 3.00000 is different with 3.35000 at 6 significant digits, with the same exponent). [Example 4-4] 3.0×10^15 and 30×10^14 are equivalent (since both can be written as 3.00000×10^15 at 6 significant digits with the same exponent). [Example 4-5] \"\frac{99\sqrt{3}}{100}\" and \"The distance between $x$ and $y$ in the original space $X$ is slightly less than $\sqrt{3}$\" are not equivalent because the former candidate is 1.71362, whereas the latter candidate only states that the distance is \"slightly less than $\sqrt{3}$\" (with $\sqrt{3}$ = 1.73205), and 1.71362 and 1.73205 are not the same.
<Rule 5> For multiple-choice questions, answers may be given as letters (e.g., A/B/C/D/E/F, (1)/(2)/(3)/(4)), as the option text itself, or a mix of both. Two candidates are considered equivalent only if they select the same option, either by letter or by exactly matching the option text (e.g., (A) and (C) should always be not equivalent). Ensure that the selected answer matches the corresponding option in the question—if the answer is given as text, verify that it aligns with the correct lettered choice in the problem. If the candidate’s answer consists only of text that does not match any of the provided options, it should be considered inequivalent, as the candidate’s response is not a valid choice. In case one of the candidate does not provide a prediction at all (e.g., an empty string), the two candidates should always be marked as not equivalent. Moreover, if one candidate made chose multiple options at once and the other candidate only chose one of the options, they are not equivalent.
<Rule 6> When there is a type mismatch between candidates—such as when one answer is a text span and the other is an approximate equality—these are not considered equivalent except when there is strong evidence that they are equivalent. In particular, if one candidate does not provide a conclusive answer (for example, by not explicitly stating the actual numerical value and instead offering only relevant context or description), do not consider the answers equivalent. Refer to rule 8 about \"lazy reasoning\" behavior, which should be considered when determining equivalence of two candidates. [Example 6-1] \"Given the linewidth of the Ne 6328 Å line observed in spontaneous emission is 0.016 Å wide, the laser would likely operate at several axial frequencies.\" and \"\frac{\Delta f'}{\Delta f} \approx 4\" are not equivalent since \"several\" and \"4\" are not equivalent. [Example 6-2] \"The generalized coordinate is $x$.\" and \"answer 1: x\" are equivalent since both candidates specify $x$. [Example 6-3] $L = \frac{1}{2} m (1 + \frac{4x^2}{a^2}) \dot{x}^2 - m g \frac{x^2}{a}$ and \"L = \frac{1}{2} m \dot{x}^2 \left(1 + \frac{4x^2}{a^2}\right) - \frac{mgx^2}{a}\" are equivalent since both expressions are algebraically identical, just written in a different order. [Example 6-4] \"(0, 0)\" and \"The equilibrium position is $x = 0$.\" are equivalent when the question asked for the equilibrium position of a particle, because both candidates identify the equilibrium position as being at x=0 (and thus z=0), which is the unique equilibrium point for the system. The slight difference in format does not omit any required information, since the value of z is determined by the constraint and does not need to be stated separately. [Example 6-5] \"The eccentricity of the planetary orbit after the explosion will be very close to 1, indicating a highly elliptical orbit.\" and \"e = \sqrt{1 + \left( \frac{M}{M'} \right)^2 \left( 1 - \frac{2M'}{M} \right)}\" are not equivalent because general or qualitative answer (an approximate statement), while the other provides the explicit, detailed result (an equation). [Example 6-6] \"\frac{d\sigma}{d\Omega} = \frac{R^2}{4}\" and \"The classical cross section for elastic scattering of point particles from an infinitely massive sphere of radius $R$ is isotropic\" are not equivalent because the former candidate provides explicit value of the differential cross section as an equation while the latter candidate only restates the qualitative property (isotropy) as a text span and does not provide the explicit value or formula for the cross section. [Example 6-7] \"p(x) = \frac{1}{\pi} \left( \frac{k}{2E - kx^2} \right)^{\frac{1}{2}}\" and \"\[ p(x) = \begin{cases} \frac{1}{2 \sqrt{\frac{2E}{k}}} & \text{if } -\sqrt{\frac{2E}{k}} \leq x \leq \sqrt{\frac{2E}{k}} \\0 & \text{otherwise} \end{cases} \]\" are not equivalent because the former candidate is probability density function for a classical harmonic oscillator with total energy E, derived from the time the oscillator spends at position x, while the latter candidate is a uniform distribution over the allowed range of x. In this case, do not simply conclude they are not equivalent because one is an equation and the other is a piecewise function, but reason why they are different. [Example 6-8] For the question \"Consider a boundary value problem in a two-dimensional domain, where the solution is highly irregular and the domain exhibits complex geometry. Which numerical method would be most suitable for solving this problem, where the solution's irregularity and the domain's complexity make it challenging to find an accurate approximation using traditional methods?\", candidate \"Yes\" and \"Monte Carlo simulation\" are not equivalent because the former is a binary decision and the latter directly names the numerical method that is most suitable for the described problem as a text span. [Example 6-9] \"2\" and \"2.0\" are equivalent since they represent the same numerical value, based on rule 4 (comparing up to 6 significant digits: 2.00000 and 2.00000 are the same). [Example 6-10] 1.23% and and 0.123 are equivalent since percentage could be converted to decimal by dividing by 100, based on rule 4 (comparing up to 6 significant digits: 0.012300 and 0.012300 are the same).
<Rule 7> In case one of the candidate provides a whole solution or thinking process instead of the final prediction, the two candidates are not equivalent. This is to prevent the confusion of determining the equivalence of the two candidates. There could be cases where the final prediction is a text span or there are one or two sentences of explanation, which you could refer to the other rules to determine equivalence. However, if one candidate provides a detailed step-by-step solution, they are not equivalent.
<Rule 8> For questions that require writing an interval, equation, piecewise function, set, ideal, or matrix, reason carefully about whether the two candidates are mathematically equivalent, even if they look different. Consider standard mathematical conventions and context. Always check if different notations represent the same mathematical object in the context of the question. Note that in case one of the candidate omits a crucial component requested by the question, the two candidates are not equivalent. Additionally, if a candidate provides only a general, qualitative, or incomplete answer (for example, by giving a vague description or a general formula instead of the explicit, detailed result required), this is considered \"lazy reasoning\" and the answers are not equivalent, even if the general answer is related to the topic. [Example 8-1] In commutative algebra, the ideal generated by a set of pairwise coprime irreducible polynomials \"Rad(⟨f⟩)=⟨f_1f_2…fs⟩\" is equivalent to the idea generated by their product \"\\( \\text\{Rad\}(\\langle f \\rangle) = \\langle f_1, f_2, \\dots, f_s \\rangle \\)\". [Example 8-2] \"\ddot{x} + \frac{2gx}{a} = 0\" and \"The equation for small oscillations about the equilibrium is $\ddot{x} = -\frac{2g}{a} x_0$.\" are not equivalent since the former gives the standard form for the equation of motion for small oscillations (ẍ ω^2 \times x=0) while the latter candidate uses \"x_0\" instead of x, denoting the initial position. [Example 8-3] \"x = A \cos \left( \sqrt{\frac{2g}{a}} t + \epsilon \right)\" and \"$x(t) = A \cos(\sqrt{\frac{2g}{a}} t + \phi)$\" are equivalent since both give the general solution, with only a difference in the symbol for the phase constant (\epsilon vs. \phi), which is not essential. [Example 8-4] \" answer 1: r = l + \frac{mg}{k} + A \cos \left(\sqrt{\frac{k}{m}} \, t + \varphi_1 \right) * answer 2: \theta = B \cos \left(\sqrt{\frac{kg}{kl + mg}} \, t + \varphi_2 \right)\" and \"$$m \ddot{r} + k (l - r) - m g \cos(\theta) = 0$$, $$m r^2 \ddot{\theta} + 2 m r \dot{r} \dot{\theta} + m g r \sin(\theta) = 0$$\" are not equivalent because the latter candidate does not provide the explicit solutions for $r(t)$ and $\theta(t)$ in the small-displacement approximation, which are required by the question and given in the former candidate. [Example 8-5] \"P = w \left( 1 + \frac{a}{g} \right) (V + v)\" and \"$mg(V + v)$\" are not equivalent since the former candidate correctly includes the effect of the elevator's acceleration (when $a$ is not 0), while the latter candidate does not. [Example 8-6] For the question: \"Two rods $AB$ and $BC$, each of length $a$ and mass $m$, are frictionlessly joined at $B$ and lie on a frictionless horizontal table. Initially the two rods (i.e. point $A, B, C$) are collinear. An impulse $\vec{P}$ is applied at point $A$ in a direction perpendicular to the line $ABC$. Find the motion of the rods immediately after the impulse is applied.\", candidate \"The motion of the rods immediately after the impulse is applied is characterized by a translational velocity of $\frac{P}{2m} t$ and an angular velocity of $\frac{3P}{ma} t$.\" and candidate \"* answer 1: \left( 0, -\frac{\overline{P}}{4m} \right) * answer 2: \left( 0, \frac{5\overline{P}}{4m} \right) * answer 3: \dot{\theta}_1 = \frac{3\overline{P}}{2ma} * answer 4: \dot{\theta}_2 = - \frac{9\overline{P}}{2ma}\" are not equivalent not because the latter candidate provides more answers (according to Rule 2, it is okay to have four answers for a question that does not have four sub-problems since the question itself could only be solved when providing all these four equations), but because the former candidate does not provide the explicit, immediate post-impulse velocities for the relevant points and both rods as required by the question. Instead, the former candidate gives only general, time-dependent expressions and omits the specific, conclusive results, which is an example of \"lazy reasoning.\" [Example 8-7] \"H = \sqrt{\left( \mathbf{p} - \frac{q\mathbf{A}}{c} \right)^2 c^2 + m_0^2 c^4\" and \"\boxed{H = \gamma m_0 c^2 + q \phi - q \mathbf{A} \cdot \frac{\mathbf{p}}{\gamma m_0}}\" are not equivalent.  The former candidate provides a standard, fully general relativistic Hamiltonian for a charged particle in electromagnetic fields, expressed in terms of the canonical momentum p, while the latter candidate is not the standard Hamiltonian in terms of the canonical momentum; it mixes the kinetic energy, potential energy, and a term involving the vector potential and velocity (or momentum), and is not generally equivalent to the reference answer. In particular, the reference answer expresses the Hamiltonian as a function of the canonical momentum, while the extracted answer involves the velocity (through γ) and is not generally valid as a Hamiltonian function of p. You should refrain from making unjustified substitutions between canonical and kinetic momentum or from assuming equivalence based on superficial algebraic similarity; always ensure the mathematical objects are compared in the correct variables and context. [Example 8-8] For the question:\"A particle of charge $e$, energy $E$, and velocity $v$ moves in a magnetic field generated by a magnetic dipole of strength $M$ located at the origin and directed along the $z$-axis. If the particle is initially in the $xy$-plane at a distance $R$ from the origin and moving radially outward, give the minimum and maximum radii it will reach (assume the orbit is bounded).\", candidate \"* answer 1: r_{\text{max}} = \frac{\alpha}{2R} \left( 1 + \sqrt{1 - \frac{4R^2}{\alpha}} \right) * answer 2: r_{\text{min}} = \frac{\alpha}{2R} \left( -1 + \sqrt{1 + \frac{4R^2}{\alpha}} \right)\" and candidate \"R_{\text{min}} = R \sqrt{1 - \frac{2eMv}{\mu_0 m c^2}}, R_{\text{max}} = R \sqrt{1 + \frac{2eMv}{\mu_0 m c^2}}\" are not equivalent because they provide different mathematical expressions for the minimum and maximum radii, with no clear algebraic or physical equivalence between the two forms. Specifically, while attempting to convert one equation to another to check the equivalence, do not assume that a parameter (like α) in one candidate is defined in terms of the physical constants in the other candidate unless the question or answer explicitly provides this definition. Also, do not attempt to \"force\" equivalence by inventing or guessing relationships between variables that are not given or justified in the problem statement or the candidates' answers. Only declare two expressions equivalent if there is a clear, explicit, and justified mapping between all variables and parameters, either in the question or in the candidates' answers. [Example 8-9] For the question \"Instability ('radioactivity') of atomic nuclei with respect to $\alpha$-particle emission is a comparatively common phenomenon among the very heavy nuclei but proton-radioactivity is virtually nonexistent. Explain, with such relevant quantitative arguments as you can muster, this striking difference.\", candidate \"E_d > 0 \text{for } \alpha\text{-decay (for } A \geq 150\text{); } -\varepsilon < 0 \text{for proton-decay}\" and \"The stability of atomic nuclei with respect to $\alpha$-particle emission is higher than that with respect to proton-radioactivity due to the higher binding energy per nucleon for $\alpha$-particles and the significant mass difference between a proton and an $\alpha$-particle. These factors make $\alpha$-particle emission more energetically favorable in heavy nuclei, leading to its common occurrence, while proton-radioactivity is virtually nonexistent.\" are not equivalent because the former candidate provides explicit quantitative criteria (in terms of decay energy and binding energy) for when \alpha-decay and proton-decay are energetically allowed or forbidden whereas the latter candidate provides a qualitative explanation, not the explicit quantitative criteria or inequalities required by the question. This is another example of lazy reasoning from the latter candidate. [Example 8-10] \"(\nabla^2 + k^2) v(\mathbf{r}) = \frac{2m}{\hbar^2} V(\mathbf{r}) e^{ikz}\" and \"\left( \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} + k^2 \right) v(r) = V(r) e^{ikz}\" are not equivalent because \"\frac{2m}{\hbar^2}\" is omitted from the latter candidate and is included in the first candidate. This is required for the correct differential equation in the first Born approximation. [Example 8-11] For the question \"Let $\\mathbf{A} = \\begin{pmatrix} 0 & 1 \\\\-1 & 0 \\end{pmatrix}$ be a 2-dimensional matrix over the real numbers. Define the matrix exponential $\\exp(\\mathbf{A}t)$ as the infinite series \\exp(\\mathbf{A}t) = \\sum_{n=0}^{\\infty} \\frac{\\mathbf{A}^n t^n}{n!}. 1. Write the first term of the series expansion $\\exp(\\mathbf{A}t)$ as $\\texttt{(first term)} = 1 + \\texttt{(first derivative)} t + \\cdots$ 2. Compute the second derivative of the function $\\exp(\\mathbf{A}t)$ with respect to $t$ at $t=0$, and express your answer as $\\texttt{(second derivative)} = \\texttt{(value)}$.\", candidate \"\text{answer 1: } 1 + \mathbf{A} t + \cdots \quad \text{answer 2: } -\mathbf{I}\" and candidate \"\\texttt{(first term)} = 1 + \\texttt{(first derivative)} t + \\cdots \\texttt{(second derivative)} = \\texttt{(value)} = \\begin{pmatrix} -1 & 0 \\\0 & -1 \\end{pmatrix}\" are equivalent. Specifically, Both the former candidate and the latter candidate express the first term of the series as 1+At+⋯, with the latter using template notation that matches the required structure. For the second derivative, the former candidate gives −I and the latter gives the explicit matrix \\begin{pmatrix} -1 & 0 \\\0 & -1 \\end{pmatrix} which are mathematically identical, making the answers equivalent for both sub-problems. [Example 8-12] For the question \"Consider a linear difference equation for the polynomial sequence $\{P_n(x)\}$, where the sequence satisfies the recurrence relation $nP_n(x) = (x^2 - a^2)P_{n-1}(x) + \lambda P_{n-2}(x)$ for some constants $a$ and $\lambda$. Assuming $P_0(x)$ and $P_1(x)$ are given polynomials, find the degree of $P_n(x)$ for any $n > 0$ using the principle of linear independence of solutions.\", candidate \"2n-1\" and \"The degree of \(P_n(x)\) for any \(n > 0\) is \(d_1 + 2(n-1)\)\" are not equivalent becauase the former candidate gives a specific degree formula, while the latter candidate gives a general formula that depends on the initial condition. It is important not to make incorrect assumptions such as d_1=1 and consider all possible valid initial conditions when determining the equivalence of the candidates. [Example 8-13] For the question \"Consider the matrix $M = cB$, where $B = \begin{pmatrix} 0 & -1 \ 1 & 0 \end{pmatrix}$ and $c$ is a real number. Let $M$ be symmetric and negative semidefinite, and let $\vec{x} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$ be a unit vector, that is, $\langle \vec{x}, \vec{x} \rangle = x_1^2 + x_2^2 = 1$. Determine an interval within which the value of $| \langle \vec{x}, M\vec{x} \rangle |$ must lie, where $ \langle \vec{x}, M\vec{x} \rangle = \vec{x}^T M \vec{x}$ is the dot product of $\vec{x}$ with $M\vec{x}$.\", candidate \"[0, 0]\" and candidate \"[0, -c]\" are not equivalent since B is skew-symmetric, the only way for M=cB to be symmetric is if c=0, which means M=0 and $\langle \vec{x}, M\vec{x} \rangle = 0$ for any unit vector x. [Example 8-14] \"K(z, w) = \begin{cases} 0 & \text{if } |z| = r \\ \text{undefined} & \text{if } |z| \neq r \end{cases}\" and \"K(z, w) = \begin{cases} 0 & \text{if } z \notin M \\ \text{non-zero function of } z \text{ and } w & \text{if } z \in M \end{cases}\" are not equivalent because the former candidate says the kernel is zero on the sphere and undefined elsewhere while the latter candidate says the kernel is zero off the sphere and nonzero on the sphere, which means that they provide contradictory piecewise conditions for K(z,w). [Example 8-15] For the question \"Suppose $(H_n)_{n \in \mathbb{N}}$ is a sequence of Hilbert spaces and $(T_n)_{n \in \mathbb{N}}$ is a sequence of bounded linear operators such that $T_n : H_n \rightarrow H_{n+1}$. Define a new sequence of Hilbert spaces $(\tilde{H}_n)_{n \in \mathbb{N}}$ where $\tilde{H}_n = H_n \times H_{n+1}$. Consider a sequence of bounded linear operators $(\tilde{T}_n)_{n \in \mathbb{N}}$ on $(\tilde{H}_n)_{n \in \mathbb{N}}$ such that $\tilde{T}_n ((x_n, x_{n+1})) = (T_n(x_n), x_{n+1})$. Evaluate $\mathbb{E}[\tilde{T}_n]$ (the expected value of $\tilde{T}_n$), given that $T_n(x_n)$ is a random operator acting on $H_n$. Use the law of total expectation to decompose this expectation into a conditional expectation and the expected value of the operator.\", candidate \"(\mathbb{E}[T_n(x_n)], x_{n+1})\" and candidate \"\mathbb{E}[\tilde{T}_n] = (\mathbb{E}[T_n(x_n)], \mathbb{E}[x_{n+1}])\" are not equivalent because the former candidate the random operator T_n to a specific vector x_n and then takes the expectation of the result, while leaving x_{n+1} unchanged whereas the latter candidate not only takes the expectation of T_n(x_n) but also takes the expectation of x_{n+1}. [Example 8-16] For the question \"The eigenvalues of the matrix $A$, where $$ A=\left[\begin{array}{rrr} 4 & 2 & 3 \\ 4 & 5 & 3 \\ 4 & 2 & 5 \end{array}\right]$$ are $5, 2, 2$. Find all corresponding eigenvectors. Enter the eigenvectors as lists separated by commas.\", candidate \"[1, -2, -\frac{4}{3}], [1, -4, 2], [1, -1, 0]\" and candidate \"\[\begin{bmatrix} -\frac{3}{4} \\ \frac{3}{2} \\ 1 \end{bmatrix}, \quad \begin{bmatrix} \frac{3}{2} \\ -3 \\ 1 \end{bmatrix}, \quad \begin{bmatrix} \frac{3}{2} \\ -3 \\ 1 \end{bmatrix}\]\" are not equivalent because the former candidate provides three distinct eigenvectors, including two linearly independent ones for the repeated eigenvalue, whereas the latter candidate repeats the same eigenvector for the repeated eigenvalue. This means the two candidates do not span the same eigenspaces for the repeated eigenvalue. [Example 8-17] \"\begin{bmatrix} z_1 \mathbb{I}_{\mathcal{H}} & 0 & \cdots & 0 \\ 0 & z_2 \mathbb{I}_{\mathcal{H}} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & z_d \mathbb{I}_{\mathcal{H}} \end{bmatrix}\" and \"The matrix representation of the action of $\\mathbb{C}^d$ on $\\mathcal{H} \\otimes H$ is the $d \\times d$ identity matrix $\\mathbb{I}_d$\" are because only the former candidate provides the correct and complete matrix representation for the action of all elements of $\\mathbb{C}^d$ on $\\mathcal{H} \\otimes H$, showing how each coordinate z_i acts as a scalar multiple of the identity on the corresponding block. In contrast, the latter candidate only gives the $d \\times d$ identity matrix, which represents the action of the specific element (1,1,…,1) and does not capture the general action of arbitrary elements of $\\mathbb{C}^d$. Therefore, Candidate 2 omits essential information about the structure of the action, making the answers not equivalent.
<Rule 9> For questions that require a text span (e.g., explaining a concept, providing examples, naming an entity), do not judge which is better; instead, focus strictly on whether both candidates contain the same essential information and reach the same level of completion. Both answers must include all information explicitly required by the question and must be equally complete (e.g., both provide a final answer, or both provide a method, as required). Extra detail is allowed, but omission of any required element means the answers are not equivalent. [Example 9-1] If candidate 1 gives only an algebraic decomposition of an integrand and candidate 2 gives the explicit final density function, these are not equivalent unless the question asks for either a method or a result. [Example 9-2] If each candidate describes a different required property (such as one mentioning coherence and the other monochromaticity for a laser), they are not equivalent unless the question only asks for one property. If both provide all required information, even if one is more detailed, they are considered equivalent. [Example 9-3] \"(a) Precession of angular momentum; (b) Circularization and decay of orbit.\" and \"The non-central component to the Earth's gravitational field will cause the satellite's orbit to become more eccentric, with the satellite moving closer to the equator at the apogee and closer to the poles at the perigee. The atmospheric drag concentrated near the perigee will cause the satellite to lose energy and move to a higher orbit, making the orbit more circular over time.\"  are not equivalent, since the former only lists effects without explanation, while latter provides the required mechanisms and details; unless the question asks only for identification of effects. [Example 9-4] \"\frac{x_3}{x_1} \in H^1_{\mathfrak{m}}(\mathcal{R}) \text{ is non-zero.}\" and \"Since \(H^1_{\mathfrak{m}}(\mathcal{R}) \neq 0\), the ring \(\mathcal{R}\) is not Cohen-Macaulay.\" are not equivalent, since only the former candidate provides the required explicit nonzero element in the local cohomology group while the latter candidate omits this element.\n Similarly, \"\frac{x_3}{x_1} \in H^1_{\mathfrak{m}}(\mathcal{R}) \text{ is non-zero.}\" and \"The ring $\mathcal{R}=\mathbb{C}[x_1,x_2,x_3]/(x_1^2x_2-x_3^2)$ is not Cohen-Macaulay because there exists a non-vanishing element in the first local cohomology group $H^1_{\mathfrak{m}}(\mathcal{R})$, where $\mathfrak{m}=(x_1,x_2,x_3)$.\" are not equivalent, since the latter candidate only asserts existence without providing the element or explanation. [Example 9-5] \"Multiply by \texttt{cos(2\pi f\_c t)}, low-pass filter.\" and \"To demodulate the signal $s(t) = m(t)\cos(2\pi f_c t)$ and recover the original message signal $m(t)$, a receiver can use an **envelope detector** or a **synchronous detector**. The process involves multiplying the received signal by a local oscillator signal $\cos(2\pi f_c t)$ and then applying a low-pass filter to extract the message signal. Both methods are effective in recovering $m(t)$.\" are equivalent since both candidates include the essential steps (multiplication by the carrier and low-pass filtering). [Example 9-6] \"\text{Frobenius algebra}\" and \"The algebra $A$ must be a symmetric algebra.\" are not equivalent because while they are related, they are not the same. Specifically, every symmetric algebra is a Frobenius algebra, but not every Frobenius algebra is symmetric. [Example 9-7] \"Vorticity zero, but quantized vortices can exist as topological defects.\" and \"The physical significance of $\omega$ in the context of superfluid dynamics is that it represents the circulation of the superfluid around the vortex line, which is quantized due to the single-valued nature of the wave function. The quantization of the vorticity is a fundamental property of superfluids and is a manifestation of the macroscopic quantum coherence of the condensate.\" are not equivalent because the former candidate lacks the physical significance discussion, and the latter candidate lacks the explicit calculation or description of the vorticity for the given wave function when comparing against each other. [Example 9-8] \"\text{answer 1: The knot } K \text{ is the unknot.}, \text{answer 2: The knot can be unknotted.}\" and \"The knot \( K \) is either the trivial knot or a torus knot.\" are not equivalent because the candidates make different predictions about the type of knot: one says only the unknot, the other says unknot or torus knot. Also, they make different predictions about the topology: one says it can be unknotted, the other allows for knots that cannot be unknotted. In this process, it is important not to make incorrect assumptions such that the virtually abelian property excludes nontrivial torus knots, and instead focus on the explicit predictions made by each candidate. [Example 9-9] \"\begin{itemize} \item Wave Theory of Light \item Fresnel Equations \item Fresnel Integrals \item Fresnel Biprism \item Fresnel Zone Plate \item Fresnel Diffraction \item Huygens-Fresnel Principle \end{itemize}\" and \"Fresnel's contributions to the history of diffraction include his development of the wave theory of light, the mathematical theory of Fresnel diffraction, the invention of the Fresnel zone plate and lens, the formulation of Fresnel equations, the introduction of Fresnel integrals, and his groundbreaking experiments on interference and diffraction. These contributions have significantly advanced our understanding of wave optics and have had a profound impact on various scientific and engineering disciplines.\" are equivalent because both answers include all the major contributions and any extra detail (such as the impact on science and engineering in the latter candidate) is allowed. The difference in format (list vs. prose) do not affect the essential completeness or correctness of the answer.
<Rule 10> If the two candidates are equivalent, output \"Final Judgment: Yes <End of Judgment>\". If not, output \"Final Judgment: No <End of Judgment>\". Most importantly, DO NOT MAKE a judgment first. Instead, first reason about whether the candidates are equivalent or not based on the specified rules above (read through all of them, not only one), and then output the final judgment.

### Reasoning: 
"""

def validate_output(text):
    """Check if the verification output contains a valid decision."""
    if "Final Judgment: Yes" in text:
        return True, "Final Judgment: Yes"
    elif "Final Judgment: No" in text:
        return True, "Final Judgment: No"
    return False, None

def create_client():
    API_key = "90679b494bad4e729238716195bced48"
    return AzureOpenAI(
        api_version="2025-02-01-preview",  # latest API version
        api_key=API_key,
        azure_endpoint=f"https://azure-services-fair-openai1-eastus2n2.azure-api.net",
    )

def process_prompt(prompt, max_tokens):
    client = create_client()
    while True:
        try:
            completion = client.chat.completions.create(
                model="o3",
                messages=[prompt],
                max_completion_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            time.sleep(5)  # Wait for 5 seconds before retrying
            print(f"Error processing prompt: {e}")
            continue

def openai_inference(prompts, max_tokens):
    results = []
    
    max_workers = 32
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_prompt = {
            executor.submit(process_prompt, prompt, max_tokens): prompt 
            for prompt in prompts
        }
        
        # Process results as they complete
        for future in tqdm(
            future_to_prompt, 
            total=len(prompts), 
            desc="Processing evaluations"
        ):
            x = future.result()
            print(x)
            results.append(x)
    
    return results

def extract_judgment(judgment_str: str) -> tuple[str, bool]:
    """Extract judgment and determine if it's correct."""
    # Add the stop token back if it's missing
    if "<End of Judgment>" not in judgment_str:
        judgment_str += " <End of Judgment>"
    
    # Check if the judgment is "Yes" (correct)
    is_correct = "Final Judgment: Yes" in judgment_str
    
    return judgment_str, is_correct

def calculate_tokens(tokenizer, text):
    """Calculate the number of tokens in a given text using the specified tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def process_benchmarks(input_file, output_file, max_tokens):
    # Initialize tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained("/datasets/pretrained-llms/Qwen3-4B")
    
    # Load the input file - check if output file exists for resuming
    if os.path.exists(output_file):
        print(f"Output file {output_file} exists. Loading from it for resuming...")
        with open(output_file, "r") as f:
            completed_items = json.load(f)
        
        # Load original data from input file
        with open(input_file, "r") as f:
            all_items = json.load(f)
        
        # Merge completed items back into all_items by matching idx
        completed_dict = {item["idx"]: item for item in completed_items}
        for idx, item in enumerate(all_items):
            if item["idx"] in completed_dict:
                all_items[idx] = completed_dict[item["idx"]]
    else:
        print(f"Loading from input file {input_file}")
        with open(input_file, "r") as f:
            all_items = json.load(f)
    
    # First, handle items using simple string comparison (AIME/GPQA) or failed to process
    aime_gpqa_processed_count = 0
    failed_to_process_count = 0
    
    for i, item in enumerate(all_items):
        # Add token count for response if not already present
        if "response_tokens" not in item and "response" in item:
            item["response_tokens"] = calculate_tokens(tokenizer, item["response"])
            all_items[i] = item
        
        if ("judgment" not in item or "is_it_correct" not in item):
            # Handle [FAILED_TO_PROCESS] cases
            if item.get("extracted_answer") == "[FAILED_TO_PROCESS]":
                item["judgment"] = ""
                item["is_it_correct"] = False
                all_items[i] = item
                failed_to_process_count += 1
    
    print(f"Processed {aime_gpqa_processed_count} items with 'AIME' or 'GPQA' in idx using simple string comparison")
    print(f"Processed {failed_to_process_count} '[FAILED_TO_PROCESS]' items")
    
    # Filter items that need OpenAI evaluation (not AIME/GPQA, not failed, don't have judgment)
    items_to_process = []
    for i, item in enumerate(all_items):
        if (item.get("extracted_answer") != "[FAILED_TO_PROCESS]" and
            ("judgment" not in item or "is_it_correct" not in item)):
            items_to_process.append((i, item))
    
    print(f"Found {len(items_to_process)} items that need OpenAI evaluation")
    
    if not items_to_process:
        print("All items already processed. Exiting.")
        return
    
    # Split into 10 sections
    num_sections = 10
    section_size = math.ceil(len(items_to_process) / num_sections)
    
    for section_idx in range(num_sections):
        start_idx = section_idx * section_size
        end_idx = min((section_idx + 1) * section_size, len(items_to_process))
        
        if start_idx >= len(items_to_process):
            break
            
        section_items = items_to_process[start_idx:end_idx]
        print(f"Processing section {section_idx + 1}/{num_sections} with {len(section_items)} items")
        
        # Prepare prompts for this section
        prompts = []
        for _, item in section_items:
            # Handle answer format
            if type(item["answer"]) == list:
                ground_truth = " ".join(item["answer"])
            else:
                ground_truth = item["answer"]
            
            # prompt_content = DETAILED_FEW_SHOT.replace("[HERE_IS_THE_QUESTION]", item["question"]).replace("[HERE_IS_THE_GROUND_TRUTH]", ground_truth).replace("[HERE_IS_THE_CANDIDATE]", item["extracted_answer"])
            
            # prompt_content = CONCISE_ZERO_SHOT.replace("[HERE_IS_THE_QUESTION]", item["question"]).replace("[HERE_IS_THE_GROUND_TRUTH]", ground_truth).replace("[HERE_IS_THE_CANDIDATE]", item["extracted_answer"])
            
            prompt_content = DETAILED_ZERO_SHOT.replace("[HERE_IS_THE_QUESTION]", item["question"]).replace("[HERE_IS_THE_GROUND_TRUTH]", ground_truth).replace("[HERE_IS_THE_CANDIDATE]", item["extracted_answer"])
            
            prompts.append({
                "role": "user",
                "content": prompt_content
            })
        
        try:
            # Run OpenAI inference for this section
            print(f"Running OpenAI inference for section {section_idx + 1}...")
            batch_outputs = openai_inference(prompts, max_tokens)
            
            # Process results and add to items
            for (original_idx, item), judgment_output in zip(section_items, batch_outputs):
                judgment, is_correct = extract_judgment(judgment_output)
                
                # Add judgment and is_it_correct
                item["judgment"] = judgment
                item["is_it_correct"] = is_correct
                
                # Update the original item in the all_items list
                all_items[original_idx] = item
            
            print(f"Successfully processed section {section_idx + 1}")
            
        except Exception as e:
            print(f"Error processing section {section_idx + 1}: {str(e)}")
            # For failed items in this section, mark them as failed
            for original_idx, item in section_items:
                if "judgment" not in item:
                    item["judgment"] = "[FAILED_TO_PROCESS]"
                if "is_it_correct" not in item:
                    item["is_it_correct"] = False
                all_items[original_idx] = item
        
        # Save progress after each section - save all items
        with open(output_file, "w") as f:
            json.dump(all_items, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(all_items)} total items to {output_file} (section {section_idx + 1} completed)")
        
        # Log the number of items processed so far
        processed_count = sum(1 for item in all_items if "judgment" in item and "is_it_correct" in item)
        print(f"Total items processed so far: {processed_count}")
    
    # Calculate benchmark-specific statistics
    benchmark_stats = {}
    for item in all_items:
        idx = item.get("idx", "")
        if "/" in idx:
            benchmark = idx.split("/")[0]
        else:
            benchmark = "unknown"
            
        if benchmark not in benchmark_stats:
            benchmark_stats[benchmark] = {"total": 0, "correct": 0}
        
        benchmark_stats[benchmark]["total"] += 1
        if item.get("is_it_correct") == True:
            benchmark_stats[benchmark]["correct"] += 1
    
    print(f"\nBenchmark-specific statistics:")
    for benchmark, stats in sorted(benchmark_stats.items()):
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{benchmark}: {stats['correct']}/{stats['total']} ({accuracy:.4f})")
    
    print(f"\nSuccessfully processed {len(items_to_process)} items and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate responses using O3 model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens to generate.")
    
    args = parser.parse_args()
    
    process_benchmarks(args.input_file, args.output_file, args.max_tokens)
    
