import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class MatrixAI:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=2)
        self.is_trained = False

    def train_prediction_model(self, data):
        if len(data) < 2:
            return "Error: Butuh minimal 2 baris data untuk pelatihan"
        X = data[:-1]
        y = data[1:]
        X_reshaped = X.reshape(X.shape[0], -1)
        y_reshaped = y.reshape(y.shape[0], -1)
        self.model.fit(X_reshaped, y_reshaped)
        self.is_trained = True
        return "Model berhasil dilatih!"

    def predict_next_row(self, last_row):
        if not self.is_trained:
            return "Error: Model belum dilatih"
        prediction = self.model.predict(last_row.reshape(1, -1))
        return np.round(prediction).astype(int).reshape(last_row.shape)

    def cluster_matrix(self, data):
        data_reshaped = data.reshape(data.shape[0], -1)
        data_scaled = self.scaler.fit_transform(data_reshaped)
        clusters = self.kmeans.fit_predict(data_scaled)
        return clusters

def clear_screen():
    # Kompatibilitas dengan Termux
    os.system('clear')

def tampilkan_menu():
    """Menampilkan menu utama"""
    print("\n=== PROGRAM OPERASI MATRIKS DENGAN AI ===")
    print("\nMenu Utama:")
    print("1.  Input Matriks")
    print("2.  Tampilkan Matriks")
    print("3.  Penjumlahan Matriks")
    print("4.  Pengurangan Matriks")
    print("5.  Perkalian Matriks")
    print("6.  Transpose Matriks")
    print("7.  Determinan Matriks")
    print("8.  Sistem Persamaan Linear")
    print("9.  Analisis AI - Prediksi Pola")
    print("10. Analisis AI - Clustering")
    print("11. Keluar")

def tampilkan_matriks(matriks, nama="", clusters=None):
    print(f"\nMatriks {nama}:")
    print("┌" + "─" * (len(matriks[0]) * 6 + 1) + "┐")
    for idx, baris in enumerate(matriks):
        print("│", end=" ")
        for elem in baris:
            print(f"{int(elem):4d}", end=" ")
        if clusters is not None:
            print("│", f"Cluster {clusters[idx]}")
        else:
            print("│")
    print("└" + "─" * (len(matriks[0]) * 6 + 1) + "┘")

def input_matriks(nama_matriks):
    clear_screen()
    print(f"=== Input Matriks {nama_matriks} ===")
    
    while True:
        try:
            baris = int(input("\nMasukkan jumlah baris: "))
            kolom = int(input("Masukkan jumlah kolom: "))
            if baris > 0 and kolom > 0:
                break
            print("Error: Jumlah baris dan kolom harus lebih dari 0!")
        except ValueError:
            print("Error: Masukkan angka yang valid!")

    matriks = []
    template = np.zeros((baris, kolom), dtype=int)
    
    for i in range(baris):
        baris_data = []
        for j in range(kolom):
            clear_screen()
            print(f"=== Input Matriks {nama_matriks} ===")
            print(f"\nUkuran: {baris}x{kolom}")
            tampilkan_matriks(template, nama_matriks)
            print(f"\nMengisi elemen baris-{i+1} kolom-{j+1}")
            
            while True:
                try:
                    elemen = int(input(f"Masukkan nilai: "))
                    baris_data.append(elemen)
                    template[i][j] = elemen
                    break
                except ValueError:
                    print("Error: Masukkan bilangan bulat yang valid!")
        
        matriks.append(baris_data)
    
    return np.array(matriks, dtype=int)

def hitung_determinan(matriks):
    """Menghitung determinan matriks"""
    try:
        return int(round(np.linalg.det(matriks)))
    except np.linalg.LinAlgError:
        return "Error: Matriks tidak memiliki determinan"

def solve_sistem_persamaan(A, b):
    """Menyelesaikan sistem persamaan linear Ax = b"""
    try:
        solusi = np.linalg.solve(A, b)
        return np.round(solusi).astype(int)
    except np.linalg.LinAlgError:
        return "Error: Sistem persamaan tidak memiliki solusi bulat"

def input_sistem_persamaan():
    """Input sistem persamaan linear"""
    print("\nMasukkan koefisien matriks A:")
    A = input_matriks("Koefisien")
    
    print("\nMasukkan vektor b (hasil persamaan):")
    b = np.array([int(input(f"b{i+1}: ")) for i in range(len(A))])
    
    return A, b

def operasi_matriks():
    matrix_ai = MatrixAI()
    A = None
    B = None
    
    while True:
        clear_screen()
        tampilkan_menu()
        
        if A is not None and B is not None:
            print("\nMatriks saat ini:")
            tampilkan_matriks(A, "A")
            tampilkan_matriks(B, "B")
        
        pilihan = input("\nPilih menu (1-11): ")
        
        if pilihan == '1':
            A = input_matriks("A")
            B = input_matriks("B")
        
        elif pilihan == '2':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                tampilkan_matriks(A, "A")
                tampilkan_matriks(B, "B")
        
        elif pilihan == '3':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                try:
                    hasil = A + B
                    print("\nHasil A + B:")
                    tampilkan_matriks(hasil, "A + B")
                except ValueError:
                    print("\nError: Dimensi matriks harus sama!")
        
        elif pilihan == '4':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                try:
                    hasil = A - B
                    print("\nHasil A - B:")
                    tampilkan_matriks(hasil, "A - B")
                except ValueError:
                    print("\nError: Dimensi matriks harus sama!")
        
        elif pilihan == '5':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                try:
                    hasil = np.dot(A, B)
                    print("\nHasil A × B:")
                    tampilkan_matriks(hasil, "A × B")
                except ValueError:
                    print("\nError: Dimensi matriks tidak sesuai untuk perkalian!")
        
        elif pilihan == '6':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                print("\nTranspose Matriks A:")
                tampilkan_matriks(A.T, "A^T")
                print("\nTranspose Matriks B:")
                tampilkan_matriks(B.T, "B^T")
        
        elif pilihan == '7':
            if A is None:
                print("\nError: Matriks belum diinput!")
            else:
                det_A = hitung_determinan(A)
                det_B = hitung_determinan(B)
                print(f"\nDeterminan Matriks A: {det_A}")
                print(f"Determinan Matriks B: {det_B}")
        
        elif pilihan == '8':
            print("\nSistem Persamaan Linear (Ax = b)")
            A_sys, b = input_sistem_persamaan()
            solusi = solve_sistem_persamaan(A_sys, b)
            if isinstance(solusi, str):
                print(solusi)
            else:
                print("\nSolusi x:")
                for i, x in enumerate(solusi):
                    print(f"x{i+1} = {x}")
        
        elif pilihan == '9':
            if A is None:
                print("\nError: Matriks belum diinput!")
            else:
                print("\nAnalisis AI - Prediksi Pola")
                print("Melatih model dengan Matriks A...")
                result = matrix_ai.train_prediction_model(A)
                print(result)
                if matrix_ai.is_trained:
                    pred = matrix_ai.predict_next_row(A[-1])
                    print("\nPrediksi baris berikutnya untuk Matriks A:")
                    tampilkan_matriks(pred, "Prediksi")
        
        elif pilihan == '10':
            if A is None or B is None:
                print("\nError: Matriks belum diinput!")
            else:
                print("\nAnalisis AI - Clustering")
                clusters_A = matrix_ai.cluster_matrix(A)
                print("\nHasil Clustering Matriks A:")
                tampilkan_matriks(A, "A (dengan cluster)", clusters_A)
                
                clusters_B = matrix_ai.cluster_matrix(B)
                print("\nHasil Clustering Matriks B:")
                tampilkan_matriks(B, "B (dengan cluster)", clusters_B)
        
        elif pilihan == '11':
            print("\nTerima kasih telah menggunakan program ini!")
            break
        
        else:
            print("\nPilihan tidak valid!")
        
        input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    try:
        operasi_matriks()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh pengguna.")
        sys.exit(0)
