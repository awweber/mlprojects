class DecisionRegionPlotter:
    import matplotlib.pyplot as plt

    @staticmethod
    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = DecisionRegionPlotter.ListedColormap(colors[:len(DecisionRegionPlotter.np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = DecisionRegionPlotter.np.meshgrid(
            DecisionRegionPlotter.np.arange(x1_min, x1_max, resolution),
            DecisionRegionPlotter.np.arange(x2_min, x2_max, resolution)
        )
        Z = classifier.predict(DecisionRegionPlotter.np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        DecisionRegionPlotter.plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        DecisionRegionPlotter.plt.xlim(xx1.min(), xx1.max())
        DecisionRegionPlotter.plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(DecisionRegionPlotter.np.unique(y)):
            DecisionRegionPlotter.plt.scatter(
                x=X[y == cl, 0], y=X[y == cl, 1],
                alpha=0.8, c=colors[idx],
                marker=markers[idx], label=cl,
                edgecolor='black'
            )

        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]
            DecisionRegionPlotter.plt.scatter(
                X_test[:, 0], X_test[:, 1],
                facecolors='none', edgecolor='black', alpha=1.0,
                linewidth=1, marker='o',
                s=100, label='test set'
            )