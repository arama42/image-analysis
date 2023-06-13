def edge_inking_recur(self, img):
    m, n = img.shape
    linked = np.zeros((m, n), dtype=np.int32)
    visited = set()

    if not img:
        return 0

    def dfs(grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return
        else:
            visited.add(set(i, j))

        grid[i][j] = '#'
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
        # diagonal neighbours
        self.dfs(grid, i - 1, j - 1)
        self.dfs(grid, i + 1, j - 1)
        self.dfs(grid, i - 1, j + 1)
        self.dfs(grid, i + 1, j - 1)

    count = 0
    for i in range(m):
        for j in range(n):
            if img[i][j] == '1':
                self.dfs(img, i, j)
                count += 1

    return count




def edge_inking(img):
    m, n = img.shape
    weak = 100
    strong = 255

    top_to_bottom = img.copy()

    for row in range(1, m):
        for col in range(1, n):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[
                    row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[
                        row - 1, col + 1] == 255 or top_to_bottom[row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = img.copy()

    for row in range(m - 1, 0, -1):
        for col in range(n - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[
                    row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[
                    row - 1, col + 1] == 255 or bottom_to_top[
                        row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = img.copy()

    for row in range(1, m):
        for col in range(n - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[
                    row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[
                    row - 1, col + 1] == 255 or right_to_left[
                        row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = img.copy()

    for row in range(m - 1, 0, -1):
        for col in range(1, n):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[
                    row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[
                    row - 1, col + 1] == 255 or left_to_right[
                        row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right
    final_image[final_image > 255] = 255
    return final_image