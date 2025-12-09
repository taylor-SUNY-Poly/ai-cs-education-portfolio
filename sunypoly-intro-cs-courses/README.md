# SUNY Polytechnic Institute – Introductory CS Courses (CS 100 & CS 108)

This directory contains selected artifacts from two introductory computer science
courses I designed and iteratively refined at SUNY Polytechnic Institute:

- **CS 100 – Introduction to Computing Seminar** (Python + CMU Graphics)
- **CS 108 – Computing Fundamentals** (C programming with a Wordle-in-the-Terminal capstone)

These courses anchor the lower-division programming experience at SUNY Poly and serve
as key on-ramps into later AI/ML work, including the Artificial Intelligence and Machine
Learning minor.

## Course Overview

### CS 100 – Introduction to Computing Seminar (Python + Graphics)

CS 100 is a first programming experience that emphasizes **engagement and visual
feedback**. Students learn fundamental Python concepts while creating interactive
graphics using the CMU Graphics library. The course sequence progresses from Python
basics and selection/repetition to event-driven programming and object-oriented designs,
with labs and modules aligned week-by-week to the schedule.

Key design elements:

- **Graphics-focused introduction**  
  Early modules use CMU Graphics to teach shapes, animation, and events, giving students
  “I built something visual” wins from week 3 onward. The interactive textbook excerpt
  (A2) demonstrates how timed animations are introduced using `onStep`, step rate
  control, and live code examples.

- **Interactive online text + Codio integration**  
  The course was migrated to Codio, replacing a traditional textbook with an instructor-
  authored, interactive notebook that mirrors in-class content. Labs, problem sets, and
  guides are all delivered through this environment, with auto graders providing immediate
  feedback.

- **Media-rich labs and projects**  
  The CMU Graphics lab sample (A3) shows a SUNY Poly–branded “DVD screensaver”
  assignment where students animate and bounce a logo around the screen, reinforcing
  movement, collision detection, and state updates. 

- **Open-ended final project with structured supports**  
  The final project and rubric (A4) scaffold design, implementation, and reflection:
  students propose a program, design the interface, implement in Python, and write a
  detailed reflection on the concepts used and future improvements. 

Planned iterations include additional scaffolding tracks for students who struggle with
open-ended design, while maintaining common learning outcomes in control flow,
functions, and core data structures.

### CS 108 – Computing Fundamentals (C + Wordle-in-the-Terminal)

CS 108 is a rigorous introduction to **C programming** with a focus on program structure,
memory, and tool fluency. The course schedule highlights a progression from basic C
syntax and top-down design to arrays, strings, recursion, structs, and dynamic data
structures, with aligned labs and problem sets throughout the semester. 

Signature elements:

- **Wordle-in-the-Terminal capstone**  
  The final project asks students to implement a full Wordle clone in the terminal using
  multiple C files, a custom `game_state_t` struct, and a dictionary of 5-letter words.
  The provided excerpt details structure design, string processing, arrays, status encoding
  for letters, and robust input validation.

- **Emphasis on modular design and testing**  
  Students implement functions in `wordle_functions.c` based on prototypes and
  documentation in a header file, and use dedicated function drivers for incremental
  testing. This mirrors professional C workflows with separate compilation and targeted
  verification. 

- **Codio migration and autograding**  
  The course was rebuilt in Codio to incorporate function-level auto graders. Students see
  failures at the function level rather than only at full-program I/O, improving debugging
  skills and concept mastery.

Together, CS 100 and CS 108 establish a coherent, scaffolded introduction to computing:
Python graphics for engagement and conceptual grounding, followed by C for systems-level
thinking and robust program design.

## Files in This Directory

- `CS 100 Selected Artifacts.pdf`  
  Includes the CS 100 course schedule, an excerpt from the interactive CMU Graphics
  textbook chapter on timed animations, a graphics-based lab problem (SUNY Poly logo
  “DVD” screensaver), and the final project instructions and rubric. 

- `CS 108 Selected Artifacts.pdf`  
  Includes the CS 108 course schedule and the full Wordle-in-the-Terminal final project
  description with its scoring rubric and implementation details. 

## Role in the Broader Curriculum

These courses:

- Provide **novice-friendly on-ramps** into computing for SUNY Poly students from a
  variety of majors.
- Feed directly into **CS 295 – Artificial Intelligence Applications** and the
  **Artificial Intelligence & Machine Learning minor**, supplying core programming
  fluency for later AI work.
- Serve as concrete, citable examples of **curriculum design, media-rich pedagogy, and
  assessment** that align with the goals of AI/CS education grants and outreach efforts.
