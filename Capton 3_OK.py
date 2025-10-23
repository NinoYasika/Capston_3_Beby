# --- Fungsi rekomendasi film dengan filter unik dan hasil berbeda ---
def get_similar_movies(title, top_k=3):
    try:
        # Ambil banyak hasil agar bisa disaring
        similar_docs = qdrant.similarity_search_with_score(title, k=top_k + 30)

        # Fungsi bantu untuk normalisasi teks
        def normalize_text(text):
            text = text.lower().strip()
            text = re.sub(r'[^a-z0-9 ]', '', text)
            text = re.sub(r'\bthe\b', '', text)  # hapus 'the' agar lebih netral
            return text.strip()

        title_norm = normalize_text(title)
        unique_titles = set()
        filtered = []

        for doc, score in similar_docs:
            raw_title = doc.metadata.get("Series_Title", "")
            movie_title = normalize_text(raw_title)

            # --- Syarat penyaringan ---
            # 1. Tidak sama dengan film utama (secara teks)
            # 2. Tidak duplikat
            # 3. Tidak terlalu mirip (skor di atas 0.97 berarti hampir sama filmnya)
            if (
                movie_title != title_norm
                and title_norm not in movie_title
                and movie_title not in title_norm
                and movie_title not in unique_titles
                and score < 0.97
            ):
                unique_titles.add(movie_title)
                filtered.append(doc)

            if len(filtered) >= top_k:
                break

        # Tambah hasil cadangan kalau kurang dari 3
        if len(filtered) < top_k:
            extras = [
                d for d, s in similar_docs
                if normalize_text(d.metadata.get("Series_Title", "")) not in unique_titles
                and normalize_text(d.metadata.get("Series_Title", "")) != title_norm
                and s < 0.97
            ]
            filtered.extend(extras[: top_k - len(filtered)])

        # Ambil hanya 3 film
        recommendations = [doc.metadata["Series_Title"] for doc in filtered[:top_k]]

        # Safety: pastikan film utama tidak ikut muncul
        recommendations = [
            r for r in recommendations if normalize_text(r) != title_norm
        ]

        # Jika masih kurang dari 3, tambahkan placeholder
        while len(recommendations) < top_k:
            recommendations.append("(Belum cukup data relevan)")

        return recommendations[:top_k]

    except Exception as e:
        return [f"Error saat mencari rekomendasi: {str(e)}"]
