
    (define (problem nqueens-problem)
      (:domain nqueens)

      (:objects 

        c_1_1 c_1_2 c_1_3 c_1_4 c_1_5 c_1_6 c_1_7 c_1_8 c_1_9 
   c_2_1 c_2_2 c_2_3 c_2_4 c_2_5 c_2_6 c_2_7 c_2_8 c_2_9 
   c_3_1 c_3_2 c_3_3 c_3_4 c_3_5 c_3_6 c_3_7 c_3_8 c_3_9 
   c_4_1 c_4_2 c_4_3 c_4_4 c_4_5 c_4_6 c_4_7 c_4_8 c_4_9 
   c_5_1 c_5_2 c_5_3 c_5_4 c_5_5 c_5_6 c_5_7 c_5_8 c_5_9 
   c_6_1 c_6_2 c_6_3 c_6_4 c_6_5 c_6_6 c_6_7 c_6_8 c_6_9 
   c_7_1 c_7_2 c_7_3 c_7_4 c_7_5 c_7_6 c_7_7 c_7_8 c_7_9 
   c_8_1 c_8_2 c_8_3 c_8_4 c_8_5 c_8_6 c_8_7 c_8_8 c_8_9 
   c_9_1 c_9_2 c_9_3 c_9_4 c_9_5 c_9_6 c_9_7 c_9_8 c_9_9 
   - cell
        q1 q2 q3 q4 q5 q6 q7 q8 q9 - queen
      )

      (:init
        (= (min_x) 1)  (= (min_y) 1)
        (= (max_x) 9)  (= (max_y) 9) 

  (queen-at q1 c_1_5) 
  (queen-at q2 c_2_3) 
  (queen-at q3 c_3_1) 
  (queen-at q4 c_4_6) 
  (queen-at q5 c_6_2) 
  (queen-at q6 c_7_4) 
  (queen-at q7 c_8_7) 

  (used q1)  (used q2)  (used q3)  (used q4)  (used q5)  (used q6)  (used q7)

  (= (x c_1_1) 1) (= (y c_1_1) 1)   (safe-diag1 c_1_1)  (safe-diag2 c_1_1)
  (= (x c_1_2) 1) (= (y c_1_2) 2)   (safe-diag2 c_1_2)
  (= (x c_1_3) 1) (= (y c_1_3) 3) 
  (= (x c_1_4) 1) (= (y c_1_4) 4)   (safe-diag1 c_1_4)
  (= (x c_1_5) 1) (= (y c_1_5) 5) 
  (= (x c_1_6) 1) (= (y c_1_6) 6)   (safe-diag1 c_1_6)  (safe-diag2 c_1_6)
  (= (x c_1_7) 1) (= (y c_1_7) 7)   (safe-diag1 c_1_7)
  (= (x c_1_8) 1) (= (y c_1_8) 8)   (safe-col c_1_8)  (safe-diag1 c_1_8)  (safe-diag2 c_1_8)
  (= (x c_1_9) 1) (= (y c_1_9) 9)   (safe-col c_1_9)  (safe-diag1 c_1_9)

  (= (x c_2_1) 2) (= (y c_2_1) 1)   (safe-diag2 c_2_1)
  (= (x c_2_2) 2) (= (y c_2_2) 2)   (safe-diag1 c_2_2)
  (= (x c_2_3) 2) (= (y c_2_3) 3) 
  (= (x c_2_4) 2) (= (y c_2_4) 4) 
  (= (x c_2_5) 2) (= (y c_2_5) 5)   (safe-diag1 c_2_5)  (safe-diag2 c_2_5)
  (= (x c_2_6) 2) (= (y c_2_6) 6) 
  (= (x c_2_7) 2) (= (y c_2_7) 7)   (safe-diag1 c_2_7)  (safe-diag2 c_2_7)
  (= (x c_2_8) 2) (= (y c_2_8) 8)   (safe-col c_2_8)  (safe-diag1 c_2_8)
  (= (x c_2_9) 2) (= (y c_2_9) 9)   (safe-col c_2_9)  (safe-diag1 c_2_9)

  (= (x c_3_1) 3) (= (y c_3_1) 1) 
  (= (x c_3_2) 3) (= (y c_3_2) 2) 
  (= (x c_3_3) 3) (= (y c_3_3) 3)   (safe-diag1 c_3_3)
  (= (x c_3_4) 3) (= (y c_3_4) 4)   (safe-diag2 c_3_4)
  (= (x c_3_5) 3) (= (y c_3_5) 5) 
  (= (x c_3_6) 3) (= (y c_3_6) 6)   (safe-diag1 c_3_6)  (safe-diag2 c_3_6)
  (= (x c_3_7) 3) (= (y c_3_7) 7) 
  (= (x c_3_8) 3) (= (y c_3_8) 8)   (safe-col c_3_8)  (safe-diag1 c_3_8)
  (= (x c_3_9) 3) (= (y c_3_9) 9)   (safe-col c_3_9)  (safe-diag1 c_3_9)  (safe-diag2 c_3_9)

  (= (x c_4_1) 4) (= (y c_4_1) 1) 
  (= (x c_4_2) 4) (= (y c_4_2) 2) 
  (= (x c_4_3) 4) (= (y c_4_3) 3)   (safe-diag2 c_4_3)
  (= (x c_4_4) 4) (= (y c_4_4) 4)   (safe-diag1 c_4_4)
  (= (x c_4_5) 4) (= (y c_4_5) 5)   (safe-diag2 c_4_5)
  (= (x c_4_6) 4) (= (y c_4_6) 6) 
  (= (x c_4_7) 4) (= (y c_4_7) 7)   (safe-diag1 c_4_7)
  (= (x c_4_8) 4) (= (y c_4_8) 8)   (safe-col c_4_8)  (safe-diag2 c_4_8)
  (= (x c_4_9) 4) (= (y c_4_9) 9)   (safe-col c_4_9)  (safe-diag1 c_4_9)  (safe-diag2 c_4_9)

  (= (x c_5_1) 5) (= (y c_5_1) 1)   (safe-row c_5_1)
  (= (x c_5_2) 5) (= (y c_5_2) 2)   (safe-row c_5_2)  (safe-diag2 c_5_2)
  (= (x c_5_3) 5) (= (y c_5_3) 3)   (safe-row c_5_3)
  (= (x c_5_4) 5) (= (y c_5_4) 4)   (safe-row c_5_4)  (safe-diag2 c_5_4)
  (= (x c_5_5) 5) (= (y c_5_5) 5)   (safe-row c_5_5)  (safe-diag1 c_5_5)
  (= (x c_5_6) 5) (= (y c_5_6) 6)   (safe-row c_5_6)
  (= (x c_5_7) 5) (= (y c_5_7) 7)   (safe-row c_5_7)  (safe-diag2 c_5_7)
  (= (x c_5_8) 5) (= (y c_5_8) 8)   (safe-row c_5_8)  (safe-col c_5_8)  (safe-diag1 c_5_8)  (safe-diag2 c_5_8)
  (= (x c_5_9) 5) (= (y c_5_9) 9)   (safe-row c_5_9)  (safe-col c_5_9)  (safe-diag2 c_5_9)

  (= (x c_6_1) 6) (= (y c_6_1) 1)   (safe-diag1 c_6_1)  (safe-diag2 c_6_1)
  (= (x c_6_2) 6) (= (y c_6_2) 2) 
  (= (x c_6_3) 6) (= (y c_6_3) 3)   (safe-diag2 c_6_3)
  (= (x c_6_4) 6) (= (y c_6_4) 4) 
  (= (x c_6_5) 6) (= (y c_6_5) 5) 
  (= (x c_6_6) 6) (= (y c_6_6) 6)   (safe-diag1 c_6_6)  (safe-diag2 c_6_6)
  (= (x c_6_7) 6) (= (y c_6_7) 7)   (safe-diag2 c_6_7)
  (= (x c_6_8) 6) (= (y c_6_8) 8)   (safe-col c_6_8)  (safe-diag2 c_6_8)
  (= (x c_6_9) 6) (= (y c_6_9) 9)   (safe-col c_6_9)  (safe-diag1 c_6_9)

  (= (x c_7_1) 7) (= (y c_7_1) 1)   (safe-diag1 c_7_1)
  (= (x c_7_2) 7) (= (y c_7_2) 2)   (safe-diag1 c_7_2)  (safe-diag2 c_7_2)
  (= (x c_7_3) 7) (= (y c_7_3) 3) 
  (= (x c_7_4) 7) (= (y c_7_4) 4) 
  (= (x c_7_5) 7) (= (y c_7_5) 5)   (safe-diag2 c_7_5)
  (= (x c_7_6) 7) (= (y c_7_6) 6)   (safe-diag2 c_7_6)
  (= (x c_7_7) 7) (= (y c_7_7) 7)   (safe-diag1 c_7_7)  (safe-diag2 c_7_7)
  (= (x c_7_8) 7) (= (y c_7_8) 8)   (safe-col c_7_8)
  (= (x c_7_9) 7) (= (y c_7_9) 9)   (safe-col c_7_9)  (safe-diag2 c_7_9)

  (= (x c_8_1) 8) (= (y c_8_1) 1)   (safe-diag1 c_8_1)  (safe-diag2 c_8_1)
  (= (x c_8_2) 8) (= (y c_8_2) 2)   (safe-diag1 c_8_2)
  (= (x c_8_3) 8) (= (y c_8_3) 3)   (safe-diag1 c_8_3)
  (= (x c_8_4) 8) (= (y c_8_4) 4)   (safe-diag2 c_8_4)
  (= (x c_8_5) 8) (= (y c_8_5) 5)   (safe-diag2 c_8_5)
  (= (x c_8_6) 8) (= (y c_8_6) 6)   (safe-diag2 c_8_6)
  (= (x c_8_7) 8) (= (y c_8_7) 7) 
  (= (x c_8_8) 8) (= (y c_8_8) 8)   (safe-col c_8_8)  (safe-diag1 c_8_8)  (safe-diag2 c_8_8)
  (= (x c_8_9) 8) (= (y c_8_9) 9)   (safe-col c_8_9)  (safe-diag2 c_8_9)

  (= (x c_9_1) 9) (= (y c_9_1) 1)   (safe-row c_9_1)  (safe-diag1 c_9_1)
  (= (x c_9_2) 9) (= (y c_9_2) 2)   (safe-row c_9_2)  (safe-diag1 c_9_2)
  (= (x c_9_3) 9) (= (y c_9_3) 3)   (safe-row c_9_3)  (safe-diag1 c_9_3)  (safe-diag2 c_9_3)
  (= (x c_9_4) 9) (= (y c_9_4) 4)   (safe-row c_9_4)  (safe-diag1 c_9_4)  (safe-diag2 c_9_4)
  (= (x c_9_5) 9) (= (y c_9_5) 5)   (safe-row c_9_5)  (safe-diag2 c_9_5)
  (= (x c_9_6) 9) (= (y c_9_6) 6)   (safe-row c_9_6)
  (= (x c_9_7) 9) (= (y c_9_7) 7)   (safe-row c_9_7)  (safe-diag2 c_9_7)
  (= (x c_9_8) 9) (= (y c_9_8) 8)   (safe-row c_9_8)  (safe-col c_9_8)  (safe-diag2 c_9_8)
  (= (x c_9_9) 9) (= (y c_9_9) 9)   (safe-row c_9_9)  (safe-col c_9_9)  (safe-diag1 c_9_9)  (safe-diag2 c_9_9)


      ) 

      (:goal
        (and
          (used q8)
          (used q9)
          ; (> (dist q8 q9) 0)
        )
      )
      (:metric minimize (dist q8 q9))
    )
    