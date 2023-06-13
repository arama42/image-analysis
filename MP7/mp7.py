import cv2
import numpy as np


class FaceTracking:
    def __init__(self, directory, output):
        self.img_array = []
        self.directory = directory
        self.output = output
        self.img = cv2.cvtColor(cv2.imread(directory + '0001.jpg'), cv2.COLOR_BGR2GRAY)

    def track(self, method='CC'):
        bb = []

        # get bounding box coordinates
        def get_bounding_box(event, x, y, flags, params):
            nonlocal bb
            display = self.img.copy()

            if event == cv2.EVENT_LBUTTONDOWN:
                bb = [(x, y)]
                print(f"p1: {x} {y}")
            elif event == cv2.EVENT_LBUTTONUP:
                bb.append((x, y))
                print(f"p2: {x} {y}")

                # display bounding box
                cv2.rectangle(display, bb[0], bb[1], (255, 192, 203), 1)
                cv2.imshow("Bounding Box", display)
                cv2.imwrite(self.output+'0001.jpg', display)

        cv2.imshow("Bounding Box", self.img)
        # select bounding box using mouse pointer
        cv2.setMouseCallback("Bounding Box", get_bounding_box)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # create a template image from the bounding box coordinates
        T = self.img[min(bb[0][1], bb[1][1]):max(bb[0][1], bb[1][1]),
            min(bb[0][0], bb[1][0]):max(bb[0][0], bb[1][0])]

        #T = self.img[bb[0][1]: bb[1][1]+1, bb[0][0]: bb[1][0]+1]
        cv2.imshow("Template image", T)
        cv2.imwrite('Template.jpg', T)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # gets the shapes of original image, template and local window
        m, n = self.img.shape
        t_m, t_n = T.shape
        window_m = m // 6
        window_n = n // 6

        print(f"m {m} n {n}")
        print(f"t_m {t_m} t_n {t_n}")
        print(f"m {m} n {n}")
        print(f"window_margin_m {window_m} window_margin_n {window_n}")

        # track object in each frame
        for im in range(1, 501):
            file = str(im).zfill(4)
            name = self.directory + file + '.jpg'
            print(f"name {name}")

            original = cv2.imread(name)
            test = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

            # create search window margins
            window_points = [(bb[0][0] - window_n, bb[0][1] - window_m),
                             (bb[1][0] + window_n, bb[1][1] + window_m)]

            print(f"window_points {window_points}")

            # max_val if CC or NCC, min_val if SSD
            max_val = float('-inf')
            min_val = float('inf')
            new_bb = [[], []]

            # local exhaustive search
            for y in range(max(0, window_points[0][1]), min(m, window_points[1][1] - 1)):
                for x in range(max(0, window_points[0][0]), min(n, window_points[1][0] - 1)):
                    if y + t_m in range(m) and x + t_n in range(n):
                        I = test[y:y + t_m, x:x + t_n]
                        if method == 'SSD':
                            val = self.SSD(I, T)
                            if min_val > val:
                                min_val = val
                                new_bb = [(x, y), (x + t_n, y + t_m)]
                                new_T = I
                        else:
                            if method == 'CC':
                                val = self.CC(I, T)
                            elif method == 'NCC':
                                val = self.NCC(I, T)

                            if max_val < val:
                                max_val = val
                                new_bb = [(x, y), (x + t_n, y + t_m)]
                                new_T = I

            # set new template and new bounding box
            T = new_T
            bb = new_bb

            # draw image boundary
            cv2.rectangle(original, bb[0], bb[1], (147, 20, 255), 1)
            # save image
            self.img_array.append(original)
            cv2.imwrite(self.output + file + '.jpg', original)
            print(im)

    # sum of squared difference
    def SSD(self, I, T):
        ssd = 0
        diff = I - T
        ssd = np.sum(diff ** 2)
        return ssd

    # cross-correlation
    def CC(self, I, T):
        cc = 0
        mult = I * T
        cc = np.sum(mult)
        return cc

    # normalized cross-correlation
    def NCC(self, I, T):
        ncc = 0
        i_mean = np.mean(I)
        t_mean = np.mean(T)

        i_norm = I - i_mean
        t_norm = T - t_mean

        cc = np.sum(i_norm * t_norm)
        norm = np.sqrt(np.sum(i_norm ** 2) * np.sum(t_norm ** 2))
        ncc = cc / norm
        return ncc

    def save_video(self, filename):
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (128, 96))
        for i in range(len(self.img_array)):
            out.write(self.img_array[i])
        out.release()

if __name__ == '__main__':
    face = FaceTracking('image_girl/', 'Results/')
    face.track('CC')
    face.save_video('output_video-CC.mp4')
