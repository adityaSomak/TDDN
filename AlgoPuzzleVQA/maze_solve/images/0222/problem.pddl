(define (problem maze-problem)
  (:domain maze)
  (:objects
    x1 x2 x3 x4 x5 x6 x7 x8 x9 - position
    y1 y2 y3 y4 y5 y6 y7 y8 y9 - position
    agent1 - agent
  )
  (:init
  (inc x1 x2)  (inc x2 x3)  (inc x3 x4)  (inc x4 x5)  (inc x5 x6)  (inc x6 x7)  (inc x7 x8)  (inc x8 x9)
  (inc y1 y2)  (inc y2 y3)  (inc y3 y4)  (inc y4 y5)  (inc y5 y6)  (inc y6 y7)  (inc y7 y8)  (inc y8 y9)
  (dec x9 x8)  (dec x8 x7)  (dec x7 x6)  (dec x6 x5)  (dec x5 x4)  (dec x4 x3)  (dec x3 x2)  (dec x2 x1)
  (dec y9 y8)  (dec y8 y7)  (dec y7 y6)  (dec y6 y5)  (dec y5 y4)  (dec y4 y3)  (dec y3 y2)  (dec y2 y1)

  (wall x1 y1)  (wall x1 y3)  (wall x1 y4)  (wall x1 y5)  (wall x1 y6)  (wall x1 y7)  (wall x1 y8)  (wall x1 y9)
  (wall x2 y1)  (wall x2 y7)  (wall x2 y9)
  (wall x3 y1)  (wall x3 y2)  (wall x3 y3)  (wall x3 y4)  (wall x3 y5)  (wall x3 y7)  (wall x3 y9)
  (wall x4 y1)  (wall x4 y5)  (wall x4 y9)
  (wall x5 y1)  (wall x5 y3)  (wall x5 y4)  (wall x5 y5)  (wall x5 y6)  (wall x5 y7)  (wall x5 y9)
  (wall x6 y1)  (wall x6 y9)
  (wall x7 y1)  (wall x7 y3)  (wall x7 y4)  (wall x7 y5)  (wall x7 y6)  (wall x7 y7)  (wall x7 y8)  (wall x7 y9)
  (wall x8 y1)
  (wall x9 y1)  (wall x9 y2)  (wall x9 y3)  (wall x9 y4)  (wall x9 y5)  (wall x9 y6)  (wall x9 y7)  (wall x9 y8)  (wall x9 y9)


  (at agent1 x1 y2)
  )
  (:goal
    (at agent1 x8 y9)
  )
)