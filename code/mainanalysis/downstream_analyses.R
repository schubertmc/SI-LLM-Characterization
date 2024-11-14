######################################################################
# Load Packages

library(dplyr)
library(strex)
library(patchwork)
library(tidyr)
library(ggplot2)
library(ggpubr)
library(viridis)
library(rstatix)
library(ComplexHeatmap)

#Set working directory
setwd("../MEDplus_hackathon/data")


# Define Color Sets for plots 
red_colors <- c("grey", "#FAA0A0", "#CD5C5C", "#8B0000", "#B22222",  "#FF0000")
category_colors <- c("#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2")
cluster_colors <- c("#c35fe2","#70b2c6", "#b8ba94", "#6693c8" )





######################################################################
# Figure 2: Columbia Suicide Severity Rating Scale (C-SSRS) scored by 
# the LLM across Reddit Posts. 
######################################################################

# Load data with C-SSRS ratings
path <- "dataset_train_10k_CSSRS_ratings_v2.csv"
data <- read.csv(path)
cssrs <- colnames(data)[str_starts(colnames(data), "CSSR")]

# Get summarized data table
data$CSSRS_freq <- as.numeric(data$CSSRS_freq)
summarized <- data %>% 
  dplyr::group_by(label)%>%
  dplyr::summarize(n = n(), 
                   CSSRS_1 = sum(CSSRS_1 == "Yes"), 
                   CSSRS_2 = sum(CSSRS_2 == "Yes"), 
                   CSSRS_3 = sum(CSSRS_3 == "Yes"), 
                   CSSRS_4 = sum(CSSRS_4 == "Yes"), 
                   CSSRS_5 = sum(CSSRS_5 == "Yes"), 
                   freq_mean = mean(CSSRS_freq, na.rm=T), 
                   freq_min = min(CSSRS_freq, na.rm=T), 
                   freq_max = max(CSSRS_freq, na.rm=T), 
                   CSSRS_any = sum(CSSRS_any), 
                   any_perc = 100 * CSSRS_any / n
  )

# Convert the data to a binary presence/absence format for the `any` column
# Create a contingency table between `label` and `any_binary`
data$any_binary <- ifelse(data$CSSRS_any > 0, 1, 0)
contingency_table <- table(data$label, data$any_binary)
# Perform the chi-squared test
chi_squared_result <- chisq.test(contingency_table)
print(chi_squared_result)



# Prepare plotting
long <- pivot_longer(data, cols = cssrs[-c(6:7)]) %>% 
  filter(value %in% c("Yes", "No")) 
long$name <- factor(long$name, levels = rev(c(cssrs[-c(6,7)])))

# Plot first part
p1 <- ggplot(long, aes(y = name, fill = value)) + 
  theme_bw() + 
  geom_bar() +
  ylab("C-SSRS Question")+
  xlab("Number of Posts")+
  facet_grid(~label, scales = "free") + 
  scale_fill_manual(values = c("grey", "red3"), name = "") + 
  theme(strip.background = element_blank(), 
        strip.text = element_text(face = "bold", 
                                  hjust = 0), 
        axis.title.x = element_blank()
  ) +
  ggtitle("Columbia-Suicide Severity Rating Scale Ratings by Subreddit")
freq <- data %>% 
  filter(CSSRS_freq >=0, 
         CSSRS_freq <=5) 
freq$CSSRS_freq <- factor(freq$CSSRS_freq, levels = (c(0,1,2,3,4,5)))
p2 <- ggplot(freq, aes(y="CSSRS Freq", fill = CSSRS_freq)) + 
  geom_bar() + 
  theme_bw()+
  scale_fill_manual(values = (red_colors), name = "")+
  facet_grid(~label, scales="free") + 
  ylab("Frequency") + 
  theme(strip.text.x = element_blank() , 
        strip.background.x = element_blank(),
        plot.margin = unit( c(0,0,0,0) , units = "lines" ) ) + 
  xlab("Number of Posts")

# Save file 
p1 / p2 + plot_layout(heights = c(5,1))
ggsave("Fig1 - Subreddit by group.png")











######################################################################
#Figure 3: Identification of features using Unsupervised Contrastive 
#Feature Identification (UCFI). 
######################################################################

# Load extracted categories
extracted_categories <- read.csv("categories_comparisons.csv")
counts <- table(extracted_categories$Category) %>% data.frame()


# Plot number of categories
p3 <- ggplot(counts %>% 
               slice_max(n=20, order_by = Freq), aes(x = Freq, y =reorder(Var1,Freq))) + 
  theme_bw() + 
  ylab("Feature Categories Identified")+
  geom_bar(stat = "identity",
           width = 0.8,
           fill = "lightblue",
           color = "black") + 
  xlab("Count (N)") + 
  ggtitle("Features identified as variable between posts")

# Plot number of features 
descriptions <- read.csv("feature_definitions_final.csv")
median(table(descriptions$category))
categories <-c("tone", "purpose", "language", "emotionalstate", "content", "mentalhealth", "relationships", "length", "request", "specifics")
descriptions$category <- factor(descriptions$category, levels = categories)
p4 <- ggplot(descriptions, aes(x =category, fill=category)) + 
  theme_bw() + 
  geom_bar() +
  scale_fill_manual(values = category_colors, name = "")+
  ylab("Number of Features defined by LLM (N)") + 
  xlab("Feature Category") + 
  theme(legend.position = "none", 
        axis.text.x = element_text(angle = 45, hjust = 1)
  )

p3 + p4 + plot_layout(widths = c(1,3))
ggsave("Fig2 - Feature Definitions.png")


# Write feature definitions 
dx <- descriptions %>%
  group_by(category) %>%
  slice_sample(n = 1)
write.csv(dx, "example_definitions.csv")










######################################################################
# Figure 4: Subtypes of Suicidal Ideation: 
# PCA,  Clustering, Marker-features
######################################################################

# Load data
feats <- "evaluated_features_f1000.csv"
feats <- read.csv(feats)
rownames(feats) <- feats$ID
feats$X <- NULL
feats$ID <- NULL
targets <- colSums(is.na(feats)) < 500


# FALSE  TRUE 
# 19   115 exclusion of features
mat <- feats[,targets]

# Replace NAs with the median value in each column
mat_na_replaced <- apply(mat, 2, function(x) {
  median_value <- median(x, na.rm = TRUE)
  x[is.na(x)] <- median_value
  return(x)
})


#------------------#
# PCA  
#------------------#

# Scale matrix columnwise
mat_sc <- scale(mat_na_replaced)

# Perform PCA with the modified matrix
pca <- prcomp(mat_sc, scale=T)
pca_df <- pca$x %>% data.frame()
pca_df$ID <- rownames(pca_df)
pca_df$group <- str_after_first(pca_df$ID, "_")


#------------------#
# K-Means  
#------------------#

# Perform elbowplot 
wss <- sapply(1:10, function(k) {
  kmeans_result <- kmeans(pca$x[,1:10], centers = k)
  kmeans_result$tot.withinss
})

plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of Clusters", 
     ylab = "Total Within-Cluster Sum of Squares")

# Set seed for reproducability
set.seed(1)
clustering <- kmeans(mat_sc, centers = 4)
pca_df$cluster <- clustering$cluster
pca_df$cluster_name <- paste("Cluster", pca_df$cluster)




#------------------#
# Visualization  
#------------------#


# Plot PCA plot 
pa <- ggplot(pca_df, aes(PC1, PC2, color = cluster_name))+
  geom_point(size =0.5 ) +
  theme_bw()+
  scale_color_manual(values = cluster_colors, name ="Cluster") 

pb <- ggplot(pca_df, aes(x= cluster, fill = group)) +
  geom_bar(position = "fill")
pa + pb
ggsave("distribution_plot.png")




# Compare it with SI Dataset
si_data <- read.csv("si_dataset_1k.csv")
filt <- si_data$ID %in% pca_df$ID
si_filt <- si_data[filt,]
cssrs <- colnames(si_filt)[str_starts(colnames(si_data), "CSSR")]
si_filt <- si_filt %>% 
  select(c(ID, cssrs))
head(si_filt)
si_filt[si_filt == "Yes"] <- 1
si_filt[si_filt == "No"] <- 0
si_filt$CSSRS_freq <- as.numeric(si_filt$CSSRS_freq)
si_filt[,2:7] <- apply(si_filt[,2:7], 2, as.numeric)
comb <- merge(pca_df, si_filt, by = "ID")
comb$CSSRS_sum <-rowSums(comb[,cssrs[-c(6:7)]])
comb$cluster_name <- paste("Cluster", comb$cluster)

# Plot sum score 
psum <- ggplot(comb, aes(x = PC1, y = PC2, color=CSSRS_sum )) + 
  theme_bw()+
  geom_point(size = 0.5)  + 
  scale_color_gradientn(colors = c("orange", "red3"), 
                        name = "C-SSRS\nSum Score")
pa + psum
ggsave("Subtypes defined clusters.png")


# Plot violin plot of suicidal ideation
pc <- ggplot(comb, aes(x = cluster_name, y=CSSRS_sum, fill = cluster_name )) + 
  theme_bw() + 
  xlab("")+
  geom_violin()  + 
  stat_summary() + 
  scale_fill_manual(name = "", 
                    values =  cluster_colors) + 
  ggtitle("C-SSRS Sum Score") + 
  ylab("C-SSRS Sum Score")  + theme(legend.position = "none")
ggsave("CSSRS_sumscore.png")




# Plot C-SSRS questions per cluster
barplotdata <- comb %>% 
  select(cluster_name, cssrs[-c(6:7)]) %>% pivot_longer(cols = cssrs[-c(6,7)])
barplotdata$value
barplotdata$value[barplotdata$value == 1] <- "Yes"
barplotdata$value[barplotdata$value == 0] <- "No"
barplotdata$name <- factor(barplotdata$name, levels = rev(c(cssrs[-c(6,7)])))

pd <- ggplot(barplotdata, aes(y =name, fill = factor(value))) +
  geom_bar(position = "fill") + 
  theme_bw() + 
  scale_fill_manual(values = c("grey", "red3"), name = "")+
  facet_grid(~cluster_name)+
  theme(strip.background = element_blank(), 
        strip.text = element_text(face = "bold", 
                                  hjust = 0), 
        axis.title.x = element_blank(),
        axis.text.x = element_text(size = 6)
  ) +
  ylab("Question")+
  ggtitle("C-SRSS Ratings by SI-Cluster")

pc + pd + plot_layout(widths = c(1, 1.5))
ggsave("subtypes_bottom.png")



#------------------#
# Identify Marker features per cluster  
#------------------#

# Prepare data with cluster and feature columns
mat_clean <- data.frame(mat_na_replaced)
mat_clean$cluster <- pca_df$cluster
data <- mat_clean
data_long <- data %>%
  pivot_longer(cols = -cluster, names_to = "feature", values_to = "value")

# Initialize an empty list to store results
marker_features_list <- list()
# Loop over each unique cluster
for (cluster_id in unique(data$cluster)) {
  print(cluster_id)
  # Create a one-vs-rest grouping variable
  data_long <- data_long %>%
    mutate(group = ifelse(cluster == cluster_id, "in_cluster", "out_cluster"))
  
  # Perform ANOVA for each feature with Bonferroni correction
  anova_results <- data_long %>%
    group_by(feature, group) %>%
    summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") %>% # Calculate means
    pivot_wider(names_from = group, values_from = mean_value, names_prefix = "mean_") %>% # Reshape to wide format
    left_join(
      data_long %>%
        group_by(feature) %>%
        anova_test(value ~ group) %>%
        data.frame() %>%
        mutate(bonferroni_threshold = 0.05 / n()), # Adjust p-value threshold
      by = "feature"
    ) %>%
    mutate(cluster = cluster_id) # Add cluster info for reference
  
  # Add results to the list
  marker_features_list[[paste("Cluster", cluster_id)]] <- anova_results
  
  # Add results to the list
  marker_features_list[[paste("Cluster", cluster_id)]] <- anova_results
}

# Combine results into a single data frame
marker_features_df_all <- bind_rows(marker_features_list)
marker_features_df <- bind_rows(marker_features_list)

# Keep significant results
marker_features_df$p.adj.sig <- marker_features_df$p < marker_features_df$bonferroni_threshold
marker_features_df <- marker_features_df %>% 
  filter(p.adj.sig)
marker_features_df$log2FC <- log2(marker_features_df$mean_in_cluster / marker_features_df$mean_out_cluster)


# Keep top 15 markers
top_markers <- marker_features_df %>%
  group_by(cluster) %>% 
  filter(log2FC>0) %>%
  slice_min(order_by = p, n = 15)
top_markers$group <- str_before_first(top_markers$feature, "_")
top_markers$cluster_name <- paste("Cluster",top_markers$cluster )
top_markers %>% 
  filter(cluster == 3)

# Rename marker columns 
top_markers$feature_name <- top_markers$feature
top_markers$feature_name <- str_replace_all(top_markers$feature_name, "_", " ")


# Plot marker features
# Cluster 1
p1 <- ggplot(top_markers %>% 
               filter(cluster == 1) %>%
               arrange(log2FC)
             , aes(x = log2FC, y = reorder(feature, log2FC), 
                   fill = group
             )) + 
  geom_bar(stat = "identity") + 
  scale_fill_manual(values = category_colors)+
  theme_bw() + 
  theme(strip.background =element_blank(), 
        strip.text = element_text(face = "bold")
  )+ ylab("")+ ggtitle("Cluster 1") 

# Cluster 2
p2 <- ggplot(top_markers %>% 
               filter(cluster == 2) %>%
               arrange(log2FC)
             , aes(x = log2FC, y = reorder(feature, log2FC), 
                   fill = group
             )) + 
  geom_bar(stat = "identity") + 
  scale_fill_manual(values = category_colors)+
  theme_bw() + 
  theme(strip.background =element_blank(), 
        strip.text = element_text(face = "bold")
  )+ ylab("")+ ggtitle("Cluster 2") 

# Cluster 3
p3 <- ggplot(top_markers %>% 
               filter(cluster == 3) %>%
               arrange(log2FC)
             , aes(x = log2FC, y = reorder(feature, log2FC), 
                   fill = group
             )) + 
  geom_bar(stat = "identity") + 
  scale_fill_manual(values = category_colors)+
  theme_bw() + 
  theme(strip.background =element_blank(), 
        strip.text = element_text(face = "bold")
  )+ ylab("")+ ggtitle("Cluster 3") 


# Cluster 4
p4 <- ggplot(top_markers %>% 
               filter(cluster == 4) %>%
               arrange(log2FC)
             , aes(x = log2FC, y = reorder(feature, log2FC), 
                   fill = group
             )) + 
  geom_bar(stat = "identity") + 
  scale_fill_manual(values = category_colors)+
  theme_bw() + 
  theme(strip.background =element_blank(), 
        strip.text = element_text(face = "bold")
  )+ ggtitle("Cluster 4") + ylab("")
ggarrange(p1,p2,p3,p4, ncol = 2, nrow =2, 
          align = "hv",
          common.legend = T, legend = "bottom")

ggsave("cluster_1-4.png")


# Save top_markers
write.csv(top_markers, "top_markers.csv")










######################################################################
# Figure 5: Feature Correlation Matrix
######################################################################

#Correlation Matrix of Features
cor_mat <- cor(mat, use="pairwise.complete.obs", 
               method = "spearman"
)


# Prepare Plot
feature_groups <- str_before_first(colnames(mat), "_")
features <- colnames(mat)
category_colors <- c(
  "tone" = "#9e0142", "purpose" = "#d53e4f", "language" = "#f46d43", 
  "emotionalstate" = "#fdae61", "content" = "#fee08b", "mentalhealth" = "#e6f598", 
  "relationships" = "#abdda4", "length" = "#66c2a5", "request" = "#3288bd", 
  "specifics" = "#5e4fa2"
)


# select cluster ids
cluster_ids <- pca_df %>% select(ID, cluster)


# Set up side bar plots for log 2 FC to identify differences across
# clusters
marker_features_df_all$log2FC <- log2(marker_features_df_all$mean_in_cluster / marker_features_df_all$mean_out_cluster)
log2FC_values <- marker_features_df_all %>% 
  select(feature, cluster, log2FC) %>% pivot_wider(names_from = cluster, 
                                                   values_from = log2FC) %>% data.frame()
log2FC_values$feature %in% rownames(cor_mat)
rownames(log2FC_values) <- log2FC_values$feature
log2FC_values <- log2FC_values[rownames(cor_mat),]
rownames(log2FC_values) == rownames(cor_mat)
log2FC_values <- log2FC_values %>% data.frame()
log2FC_values$X1 <- as.numeric(log2FC_values$X1)
log2FC_values$X2 <- as.numeric(log2FC_values$X2)
log2FC_values$X3 <- as.numeric(log2FC_values$X3)
log2FC_values$X4 <- as.numeric(log2FC_values$X4)
log2FC_values$X4[log2FC_values$X4 == -Inf] <- 0


# Setup column annotation 
row_anno <- rowAnnotation(
  Category = feature_groups,
  Subtype1 = anno_barplot(log2FC_values$X1, gp=gpar(fill = "#c35fe2")),
  Subtype2 = anno_barplot(log2FC_values$X2, gp=gpar(fill = "#70b2c6")),
  Subtype3 = anno_barplot(log2FC_values$X3, gp=gpar(fill = "#b8ba94")),
  Subtype4 = anno_barplot(log2FC_values$X4, gp=gpar(fill = "#6693c8")),
  annotation_name_gp = gpar(fontsize =6),
  annotation_name_side = "top",
  col = list(Category = category_colors)
  
  
)


# Plot and safe file 
rownames(cor_mat) <- str_replace_all(str_after_first(rownames(cor_mat), "_"), "_", " ")
pdf("correlation_heatmap.pdf", height = 10, width = 8)
pdf1 <- 
  Heatmap(cor_mat, 
          cluster_rows = T, 
          cluster_columns = T, 
          show_column_names = F, 
          row_names_gp = gpar(fontsize = 6), 
          name = "Correlation\n[ r ]",
          row_split = 4, 
          column_split = 4,
          column_title =" ",row_title = " ",
          col = viridis(10, option = "rocket"), 
          right_annotation = row_anno
  )

draw(pdf1)
dev.off()


