library(ggplot2)
df <- data.frame(
  category = rep(categories, 4),
  value = c(duet, pdf, fits, ttransformer),
  model = rep(c("DUET", "PDF", "FITS", "TTransformer"), each = length(categories))
)

ggplot(df, aes(x = category, y = value, group = model, color = model)) +
  geom_polygon(fill = NA, size = 1) +
  coord_polar() +
  theme_minimal()