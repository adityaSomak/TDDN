(define (problem maze-problem)
  (:domain maze)
  (:objects
    x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 - position
    y1 y2 y3 y4 y5 y6 y7 - position
    agent1 - agent
  )
  (:init
  (inc x1 x2)  (inc x2 x3)  (inc x3 x4)  (inc x4 x5)  (inc x5 x6)  (inc x6 x7)  (inc x7 x8)  (inc x8 x9)  (inc x9 x10)  (inc x10 x11)  (inc x11 x12)  (inc x12 x13)
  (inc y1 y2)  (inc y2 y3)  (inc y3 y4)  (inc y4 y5)  (inc y5 y6)  (inc y6 y7)
  (dec x13 x12)  (dec x12 x11)  (dec x11 x10)  (dec x10 x9)  (dec x9 x8)  (dec x8 x7)  (dec x7 x6)  (dec x6 x5)  (dec x5 x4)  (dec x4 x3)  (dec x3 x2)  (dec x2 x1)
  (dec y7 y6)  (dec y6 y5)  (dec y5 y4)  (dec y4 y3)  (dec y3 y2)  (dec y2 y1)

  (wall x1 y1)  (wall x1 y3)  (wall x1 y4)  (wall x1 y5)  (wall x1 y6)  (wall x1 y7)
  (wall x2 y1)  (wall x2 y3)  (wall x2 y7)
  (wall x3 y1)  (wall x3 y3)  (wall x3 y5)  (wall x3 y6)  (wall x3 y7)
  (wall x4 y1)  (wall x4 y3)  (wall x4 y7)
  (wall x5 y1)  (wall x5 y3)  (wall x5 y4)  (wall x5 y5)  (wall x5 y7)
  (wall x6 y1)  (wall x6 y5)  (wall x6 y7)
  (wall x7 y1)  (wall x7 y2)  (wall x7 y3)  (wall x7 y5)  (wall x7 y7)
  (wall x8 y1)  (wall x8 y3)  (wall x8 y7)
  (wall x9 y1)  (wall x9 y3)  (wall x9 y4)  (wall x9 y5)  (wall x9 y7)
  (wall x10 y1)  (wall x10 y3)  (wall x10 y7)
  (wall x11 y1)  (wall x11 y3)  (wall x11 y5)  (wall x11 y6)  (wall x11 y7)
  (wall x12 y1)  (wall x12 y7)
  (wall x13 y1)  (wall x13 y2)  (wall x13 y3)  (wall x13 y4)  (wall x13 y5)  (wall x13 y7)


  (at agent1 x1 y2)
  )
  (:goal
    (at agent1 x13 y6)
  )
)